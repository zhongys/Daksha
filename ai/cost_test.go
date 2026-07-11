package ai

import "testing"

func TestCalculateCost(t *testing.T) {
	// DeepSeek-style price sheet: ¥4/MTok input, ¥16/MTok output,
	// ¥0.8/MTok cache read → 4000 / 16000 / 800 nano-yuan per token.
	model := Model{
		Pricing: Pricing{
			Input:      4000,
			Output:     16000,
			CacheRead:  800,
			CacheWrite: 2000,
		},
	}
	usage := Usage{
		Input:        1200,
		Output:       350,
		CacheRead:    8000,
		CacheWrite:   100,
		CacheWrite1h: 50,
	}

	got := CalculateCost(model, usage)
	want := Cost{
		Input:      1200 * 4000,
		Output:     350 * 16000,
		CacheRead:  8000 * 800,
		CacheWrite: 150 * 2000,
	}
	want.Total = want.Input + want.Output + want.CacheRead + want.CacheWrite

	if got != want {
		t.Fatalf("CalculateCost = %+v, want %+v", got, want)
	}
}

func TestCalculateCostZeroPricing(t *testing.T) {
	got := CalculateCost(Model{}, Usage{Input: 1000, Output: 1000})
	if got != (Cost{}) {
		t.Fatalf("zero pricing should yield zero cost, got %+v", got)
	}
}
