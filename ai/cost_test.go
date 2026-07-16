package ai

import (
	"math"
	"math/big"
	"testing"
)

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

func TestCalculateCostSaturatesOverflow(t *testing.T) {
	model := Model{Pricing: Pricing{
		Input: math.MaxInt64, Output: math.MaxInt64,
		CacheRead: math.MaxInt64, CacheWrite: math.MaxInt64,
	}}
	usage := Usage{
		Input: 2, Output: 2, CacheRead: 2,
		CacheWrite: math.MaxInt64, CacheWrite1h: 1,
	}
	got := CalculateCost(model, usage)
	if got.Input != math.MaxInt64 || got.Output != math.MaxInt64 ||
		got.CacheRead != math.MaxInt64 || got.CacheWrite != math.MaxInt64 ||
		got.Total != math.MaxInt64 {
		t.Fatalf("overflowing cost must saturate: %+v", got)
	}
}

func TestSaturatingInt64ArithmeticMatchesBigInt(t *testing.T) {
	values := []int64{
		math.MinInt64, math.MinInt64 + 1, -1_000_000, -2, -1, 0, 1, 2,
		1_000_000, math.MaxInt64 - 1, math.MaxInt64,
	}
	min := big.NewInt(math.MinInt64)
	max := big.NewInt(math.MaxInt64)
	clamp := func(value *big.Int) int64 {
		switch {
		case value.Cmp(max) > 0:
			return math.MaxInt64
		case value.Cmp(min) < 0:
			return math.MinInt64
		default:
			return value.Int64()
		}
	}

	for _, a := range values {
		for _, b := range values {
			wantAdd := clamp(new(big.Int).Add(big.NewInt(a), big.NewInt(b)))
			if got := saturatingAddInt64(a, b); got != wantAdd {
				t.Fatalf("saturatingAddInt64(%d, %d) = %d, want %d", a, b, got, wantAdd)
			}
			wantMul := clamp(new(big.Int).Mul(big.NewInt(a), big.NewInt(b)))
			if got := saturatingMulInt64(a, b); got != wantMul {
				t.Fatalf("saturatingMulInt64(%d, %d) = %d, want %d", a, b, got, wantMul)
			}
		}
	}
}
