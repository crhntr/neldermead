package neldermead

import (
	"math"
	"testing"
)

func TestRun(t *testing.T) {
	// Define the objective function to optimize
	objective := func(x []float64) float64 {
		return math.Pow(x[0]-1, 2) + math.Pow(x[1]-2, 2)
	}

	// Define the starting point and Constraints
	x := []float64{0.0, 0.0}
	constraints := [][]float64{{-1.0, -1.0}, {2.0, 3.0}}

	// Set the options for the optimizer
	options := Options{
		MaxIterations: 100,
		Tolerance:     1e-6,
		Alpha:         1.0,
		Beta:          0.5,
		Gamma:         2.0,
		Delta:         0.5,
		Constraints:   constraints,
	}

	// Run the optimizer
	result, err := Run(objective, x, options)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// Check that the result is within the feasible region
	err = checkConstraints(result.X, constraints)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// Check that the objective value is close to the true minimum
	trueMin := 2.0
	if math.Abs(result.F-trueMin) > options.Tolerance {
		t.Errorf("unexpected objective value: got %v, want %v", result.F, trueMin)
	}
}

func FuzzRun_quadratic(f *testing.F) {
	f.Add(0.0, 0.0, -1.0, -1.0, 2.0, 3.0, 1.0, 2.0, -1.0, -2.0)
	f.Fuzz(func(t *testing.T, x1, x2, c1, c2, c3, c4, m1, m2, e1, e2 float64) {
		// Define the objective function to optimize
		objective := func(x []float64) float64 {
			return math.Pow(x[0]+m1, e1) + math.Pow(x[1]+m2, e2)
		}

		// Define the starting point and Constraints
		x := []float64{x1, x2}
		constraints := [][]float64{{c1, c2}, {c3, c4}}

		// Set the options for the optimizer
		options := Options{
			MaxIterations: 100,
			Tolerance:     1e-6,
			Alpha:         1.0,
			Beta:          0.5,
			Gamma:         2.0,
			Delta:         0.5,
			Constraints:   constraints,
		}

		// Run the optimizer
		result, err := Run(objective, x, options)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		// Check that the result is within the feasible region
		err = checkConstraints(result.X, constraints)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	})
}
