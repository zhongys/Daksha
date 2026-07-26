package ai

import "math"

// Cost is the derived price of one turn in nano-yuan (10⁻⁹ CNY).
// Tokens times per-token pricing is the source of truth for billing; Cost is
// computed from them with pure integer arithmetic and is always reproducible.
type Cost struct {
	Input      int64 `json:"input"`
	Output     int64 `json:"output"`
	CacheRead  int64 `json:"cacheRead"`
	CacheWrite int64 `json:"cacheWrite"`
	Total      int64 `json:"total"`
}

type Usage struct {
	Input        int64 `json:"input"`
	Output       int64 `json:"output"`
	CacheRead    int64 `json:"cacheRead"`
	CacheWrite   int64 `json:"cacheWrite"`
	CacheWrite1h int64 `json:"cacheWrite1h"`
	Reasoning    int64 `json:"reasoning,omitempty"`
	TotalTokens  int64 `json:"totalTokens"`
	Cost         Cost  `json:"cost"`
}

// Pricing is a model's per-token price in nano-yuan (10⁻⁹ CNY).
//
// Vendor price sheets quote yuan per million tokens; the conversion is
// ¥N/MTok → N×1000 nano-yuan/token (e.g. ¥8/MTok → 8000). Prices quoted in
// other currencies are converted to CNY when the catalog entry is written —
// the exchange rate never enters this layer.
type Pricing struct {
	Input      int64 `json:"input"`
	Output     int64 `json:"output"`
	CacheRead  int64 `json:"cacheRead"`
	CacheWrite int64 `json:"cacheWrite"`
}

// CalculateCost derives the price of one turn from token counts and the
// model's per-token pricing. Pure integer multiplication — no division, no
// rounding — so billing is reproducible to the nano-yuan. Reasoning tokens
// are already included in Output and are not billed separately. Values that
// exceed int64 saturate instead of wrapping into a bogus negative charge.
func CalculateCost(model Model, usage Usage) Cost {
	p := model.Pricing
	c := Cost{
		Input:      saturatingMulInt64(usage.Input, p.Input),
		Output:     saturatingMulInt64(usage.Output, p.Output),
		CacheRead:  saturatingMulInt64(usage.CacheRead, p.CacheRead),
		CacheWrite: saturatingMulInt64(saturatingAddInt64(usage.CacheWrite, usage.CacheWrite1h), p.CacheWrite),
	}
	c.Total = saturatingAddInt64(
		saturatingAddInt64(c.Input, c.Output),
		saturatingAddInt64(c.CacheRead, c.CacheWrite),
	)
	return c
}

func saturatingAddInt64(a, b int64) int64 {
	sum := a + b
	if b > 0 && sum < a {
		return math.MaxInt64
	}
	if b < 0 && sum > a {
		return math.MinInt64
	}
	return sum
}

func saturatingMulInt64(a, b int64) int64 {
	switch {
	case a == 0 || b == 0:
		return 0
	case a > 0 && b > 0 && a > math.MaxInt64/b:
		return math.MaxInt64
	case a > 0 && b < 0 && b < math.MinInt64/a:
		return math.MinInt64
	case a < 0 && b > 0 && a < math.MinInt64/b:
		return math.MinInt64
	case a < 0 && b < 0 && a < math.MaxInt64/b:
		return math.MaxInt64
	default:
		return a * b
	}
}
