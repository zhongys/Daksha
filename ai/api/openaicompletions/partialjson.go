package openaicompletions

import (
	"encoding/json"
	"errors"
	"io"
	"strings"
)

// parseStreamingJSON parses a potentially incomplete JSON object fragment
// accumulated during streaming tool call deltas. It always returns a usable
// map, never an error. Port of pi-ai's parseStreamingJson: strict parse,
// then strict parse of the repaired string, then partial parse, then partial
// parse of the repaired string, then give up with an empty map.
func parseStreamingJSON(s string) map[string]any {
	if strings.TrimSpace(s) == "" {
		return map[string]any{}
	}

	for _, candidate := range []string{s, repairJSON(s)} {
		if strict, err := decodeJSONObject(candidate); err == nil {
			return strict
		}
	}
	for _, candidate := range []string{s, repairJSON(s)} {
		if v, err := parsePartialJSON(candidate); err == nil {
			if m, ok := v.(map[string]any); ok {
				return m
			}
		}
	}
	return map[string]any{}
}

// decodeJSONObject performs a complete, non-lossy decode of one JSON object.
// UseNumber is important for tool arguments: the default interface decoder
// turns every number into float64, corrupting integers above 2^53 before the
// typed tool adapter gets a chance to decode them. A second decode preserves
// json.Unmarshal's strict rejection of trailing JSON values or garbage.
func decodeJSONObject(s string) (map[string]any, error) {
	decoder := json.NewDecoder(strings.NewReader(s))
	decoder.UseNumber()

	var object map[string]any
	if err := decoder.Decode(&object); err != nil {
		return nil, err
	}
	if object == nil {
		return nil, errors.New("openai: tool arguments must be a JSON object")
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			return nil, errors.New("openai: tool arguments contain multiple JSON values")
		}
		return nil, err
	}
	return object, nil
}

var validJSONEscapes = map[byte]bool{
	'"': true, '\\': true, '/': true, 'b': true, 'f': true, 'n': true, 'r': true, 't': true, 'u': true,
}

// repairJSON fixes malformed string literals: raw control characters inside
// strings are escaped, and backslashes before invalid escape characters are
// doubled. Everything outside string literals passes through untouched.
func repairJSON(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	inString := false

	for i := 0; i < len(s); i++ {
		c := s[i]

		if !inString {
			b.WriteByte(c)
			if c == '"' {
				inString = true
			}
			continue
		}

		if c == '"' {
			b.WriteByte(c)
			inString = false
			continue
		}

		if c == '\\' {
			if i+1 >= len(s) {
				b.WriteString(`\\`)
				continue
			}
			next := s[i+1]
			if next == 'u' {
				if i+6 <= len(s) && isHex(s[i+2:i+6]) {
					b.WriteString(s[i : i+6])
					i += 5
					continue
				}
				// Invalid \u escape: double the backslash, keep the rest.
				b.WriteString(`\\`)
				continue
			}
			if validJSONEscapes[next] {
				b.WriteByte('\\')
				b.WriteByte(next)
				i++
				continue
			}
			b.WriteString(`\\`)
			continue
		}

		if c <= 0x1f {
			b.WriteString(escapeControlChar(c))
			continue
		}
		b.WriteByte(c)
	}
	return b.String()
}

func isHex(s string) bool {
	for i := 0; i < len(s); i++ {
		c := s[i]
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f' || c >= 'A' && c <= 'F') {
			return false
		}
	}
	return len(s) == 4
}

func escapeControlChar(c byte) string {
	switch c {
	case '\b':
		return `\b`
	case '\f':
		return `\f`
	case '\n':
		return `\n`
	case '\r':
		return `\r`
	case '\t':
		return `\t`
	default:
		const hex = "0123456789abcdef"
		return `\u00` + string(hex[c>>4]) + string(hex[c&0xf])
	}
}

// parsePartialJSON is a recursive-descent JSON parser that tolerates
// truncation anywhere: incomplete strings yield their decoded prefix,
// incomplete numbers their valid prefix, literal prefixes (tr, fals, nul)
// their full value, and unclosed containers close implicitly. Object entries
// truncated before their value starts are dropped. Structural errors that
// are not truncation still fail.
func parsePartialJSON(s string) (any, error) {
	p := &partialParser{s: s}
	p.skipWS()
	v, err := p.parseValue()
	if err == errTruncated {
		return nil, errTruncated
	}
	return v, err
}

type jsonSyntaxError struct{ msg string }

func (e *jsonSyntaxError) Error() string { return "openai: partial json: " + e.msg }

// errTruncated means the input ended before a value even started.
var errTruncated = &jsonSyntaxError{msg: "truncated before value"}

type partialParser struct {
	s string
	i int
}

func (p *partialParser) eof() bool { return p.i >= len(p.s) }

func (p *partialParser) skipWS() {
	for !p.eof() {
		switch p.s[p.i] {
		case ' ', '\t', '\n', '\r':
			p.i++
		default:
			return
		}
	}
}

func (p *partialParser) parseValue() (any, error) {
	if p.eof() {
		return nil, errTruncated
	}
	switch c := p.s[p.i]; {
	case c == '{':
		return p.parseObject()
	case c == '[':
		return p.parseArray()
	case c == '"':
		return p.parseString()
	case c == 't', c == 'f', c == 'n':
		return p.parseLiteral()
	case c == '-' || (c >= '0' && c <= '9'):
		return p.parseNumber()
	default:
		return nil, &jsonSyntaxError{msg: "unexpected character " + string(c)}
	}
}

func (p *partialParser) parseObject() (any, error) {
	obj := map[string]any{}
	p.i++ // consume '{'

	for {
		p.skipWS()
		if p.eof() {
			return obj, nil
		}
		if p.s[p.i] == '}' {
			p.i++
			return obj, nil
		}
		if p.s[p.i] != '"' {
			return nil, &jsonSyntaxError{msg: "expected object key"}
		}

		keyStart := p.i
		key, err := p.parseString()
		if err != nil {
			return nil, err
		}
		if p.eof() && !p.stringClosed(keyStart) {
			// Truncated mid-key: drop the pair.
			return obj, nil
		}

		p.skipWS()
		if p.eof() {
			// Truncated before ':': drop the pair.
			return obj, nil
		}
		if p.s[p.i] != ':' {
			return nil, &jsonSyntaxError{msg: "expected ':' after object key"}
		}
		p.i++

		p.skipWS()
		val, err := p.parseValue()
		if err == errTruncated {
			// Truncated before the value started: drop the pair.
			return obj, nil
		}
		if err != nil {
			return nil, err
		}
		obj[key.(string)] = val

		p.skipWS()
		if p.eof() {
			return obj, nil
		}
		switch p.s[p.i] {
		case ',':
			p.i++
		case '}':
			p.i++
			return obj, nil
		default:
			return nil, &jsonSyntaxError{msg: "expected ',' or '}' in object"}
		}
	}
}

func (p *partialParser) parseArray() (any, error) {
	arr := []any{}
	p.i++ // consume '['

	for {
		p.skipWS()
		if p.eof() {
			return arr, nil
		}
		if p.s[p.i] == ']' {
			p.i++
			return arr, nil
		}

		val, err := p.parseValue()
		if err == errTruncated {
			return arr, nil
		}
		if err != nil {
			return nil, err
		}
		arr = append(arr, val)

		p.skipWS()
		if p.eof() {
			return arr, nil
		}
		switch p.s[p.i] {
		case ',':
			p.i++
		case ']':
			p.i++
			return arr, nil
		default:
			return nil, &jsonSyntaxError{msg: "expected ',' or ']' in array"}
		}
	}
}

// stringClosed reports whether the string literal starting at `start` had a
// closing quote (used to distinguish complete keys from truncated ones).
func (p *partialParser) stringClosed(start int) bool {
	escaped := false
	for i := start + 1; i < len(p.s); i++ {
		if escaped {
			escaped = false
			continue
		}
		switch p.s[i] {
		case '\\':
			escaped = true
		case '"':
			return true
		}
	}
	return false
}

func (p *partialParser) parseString() (any, error) {
	var b strings.Builder
	p.i++ // consume '"'

	for !p.eof() {
		c := p.s[p.i]
		switch c {
		case '"':
			p.i++
			return b.String(), nil
		case '\\':
			if p.i+1 >= len(p.s) {
				// Truncated mid-escape: drop it.
				p.i = len(p.s)
				return b.String(), nil
			}
			next := p.s[p.i+1]
			switch next {
			case '"', '\\', '/':
				b.WriteByte(next)
				p.i += 2
			case 'b':
				b.WriteByte('\b')
				p.i += 2
			case 'f':
				b.WriteByte('\f')
				p.i += 2
			case 'n':
				b.WriteByte('\n')
				p.i += 2
			case 'r':
				b.WriteByte('\r')
				p.i += 2
			case 't':
				b.WriteByte('\t')
				p.i += 2
			case 'u':
				if p.i+6 > len(p.s) {
					// Truncated mid-unicode-escape: drop it.
					p.i = len(p.s)
					return b.String(), nil
				}
				hexDigits := p.s[p.i+2 : p.i+6]
				if !isHex(hexDigits) {
					return nil, &jsonSyntaxError{msg: "invalid \\u escape"}
				}
				var r rune
				for j := 0; j < 4; j++ {
					r = r*16 + rune(hexVal(hexDigits[j]))
				}
				b.WriteRune(r)
				p.i += 6
			default:
				return nil, &jsonSyntaxError{msg: "invalid escape character"}
			}
		default:
			b.WriteByte(c)
			p.i++
		}
	}
	// Truncated before the closing quote.
	return b.String(), nil
}

func hexVal(c byte) int {
	switch {
	case c >= '0' && c <= '9':
		return int(c - '0')
	case c >= 'a' && c <= 'f':
		return int(c-'a') + 10
	default:
		return int(c-'A') + 10
	}
}

func (p *partialParser) parseLiteral() (any, error) {
	rest := p.s[p.i:]
	for lit, val := range map[string]any{"true": true, "false": false, "null": nil} {
		if strings.HasPrefix(rest, lit) {
			p.i += len(lit)
			return val, nil
		}
		// A strict prefix at EOF uniquely identifies the literal.
		if strings.HasPrefix(lit, rest) {
			p.i = len(p.s)
			return val, nil
		}
	}
	return nil, &jsonSyntaxError{msg: "invalid literal"}
}

func (p *partialParser) parseNumber() (any, error) {
	start := p.i
	for !p.eof() {
		c := p.s[p.i]
		if c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E' || (c >= '0' && c <= '9') {
			p.i++
			continue
		}
		break
	}
	token := p.s[start:p.i]

	// Trim trailing characters until the token parses (handles truncation
	// like "12." or "1e" or "-").
	for len(token) > 0 {
		if json.Valid([]byte(token)) {
			return json.Number(token), nil
		}
		token = token[:len(token)-1]
	}
	return nil, errTruncated
}
