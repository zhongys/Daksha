package ai

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
// are already included in Output and are not billed separately.
func CalculateCost(model Model, usage Usage) Cost {
	p := model.Pricing
	c := Cost{
		Input:      usage.Input * p.Input,
		Output:     usage.Output * p.Output,
		CacheRead:  usage.CacheRead * p.CacheRead,
		CacheWrite: (usage.CacheWrite + usage.CacheWrite1h) * p.CacheWrite,
	}
	c.Total = c.Input + c.Output + c.CacheRead + c.CacheWrite
	return c
}
