package neldermead

import (
	"math"
	"testing"
)

func TestRun(t *testing.T) {
	// Define the objective function to optimize
	objective := func(x []float64) float64 {
		return x[0] - x[1]
	}

	// Define the starting point and Constraints
	x := []float64{0, .5}
	constraints := []Constraint{
		{Min: 0, Max: 10},
		{Min: 0, Max: 10},
	}

	// Set the options for the optimizer
	options := NewOptions()
	options.Constraints = constraints

	// Run the optimizer
	result, err := Run(objective, x, options)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	requireXToBeWithinConstraints(t, result.X, constraints)

	// Check that the objective value is close to the true minimum
	trueMin := -10.0
	if math.Abs(result.F-trueMin) > options.Tolerance {
		t.Errorf("unexpected objective value: got %v, want %v (tolerance %v)", result.F, trueMin, options.Tolerance)
	}
}

func FuzzRun_quadratic(f *testing.F) {
	f.Add(0.0, 0.0, -1.0, -1.0, 2.0, 3.0, 1.0, 2.0, -1.0, -2.0)
	f.Fuzz(func(t *testing.T, xInitial1, xInitial2, min1, max1, min2, max2, m1, m2, exponent1, exponent2 float64) {
		// Define the objective function to optimize
		objective := func(x []float64) float64 {
			return math.Pow(x[0]+m1, exponent1) + math.Pow(x[1]+m2, exponent2)
		}

		// Define the starting point and Constraints
		x := []float64{xInitial1, xInitial2}
		constraints := []Constraint{{Min: min1, Max: max1}, {Min: min2, Max: max2}}

		// Set the options for the optimizer
		options := NewOptions()
		options.Constraints = constraints

		// Run the optimizer
		result, err := Run(objective, x, options)
		if err != nil {
			return
		}
		// Check that the result is within the feasible region
		requireXToBeWithinConstraints(t, result.X, constraints)
	})
}

func requireXToBeWithinConstraints(t *testing.T, x []float64, constraints []Constraint) {
	t.Helper()
	for i := range x {
		if x[i] < constraints[i].Min {
			t.Errorf("x[i]=%f is less than min constraint", x[i])
		}
		if x[i] > constraints[i].Max {
			t.Errorf("x[i]=%f is greater than max constraint", x[i])
		}
	}
}

func BenchmarkRun(b *testing.B) {
	objective := func(x []float64) float64 {
		return x[0] * x[1]
	}

	// Define the starting point and Constraints
	x := []float64{0, .5}
	constraints := []Constraint{
		{Min: 0, Max: 10},
		{Min: 0, Max: 10},
	}

	// Set the options for the optimizer
	options := NewOptions()
	options.Constraints = constraints

	for n := 0; n < b.N; n++ {
		_, err := Run(objective, x, options)
		if err != nil {
			b.Errorf("unexpected error: %v", err)
		}
	}
}
