package neldermead

import (
	"math"
	"testing"
)

func TestRun(t *testing.T) {
	t.Run("difference objective function", func(t *testing.T) {
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

		expectPoint(t, Point{F: -10, X: []float64{0, 10}}, result, 4)
	})

	t.Run("sum of squares with offset", func(t *testing.T) {
		// Define the objective function to optimize
		objective := func(x []float64) float64 {
			return math.Pow(x[0]-2, 2) + math.Pow(x[1]-3, 2) - 6
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

		expectPoint(t, Point{F: -6.0, X: []float64{2, 3}}, result, 2)
	})
}

func FuzzRun_quadratic(f *testing.F) {
	f.Add(0.0, 0.0, -1.0, -1.0, 2.0, 3.0, 1.0, 2.0, -1.0, -2.0)
	f.Add(0.0, 3.0, -1.0, -1.0, 2.0, 3.0, 1.0, 2.0, -1.0, -2.0)
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

func expectPoint(t *testing.T, exp, got Point, decimalAccuracy int) {
	t.Helper()
	diff := math.Pow10(-decimalAccuracy)
	if math.Abs(got.F-exp.F) > diff {
		t.Errorf("expected f(x0...xn) = %.[3]*[1]f got %.[3]*[2]f", exp.F, got.F, decimalAccuracy)
	}
	if len(exp.X) != len(got.X) {
		t.Errorf("expected len(Point.X) to be %d but got %d", len(exp.X), len(got.X))
	}
	if t.Failed() {
		return
	}
	for i := range exp.X {
		expX := exp.X[i]
		gotX := got.X[i]
		if math.Abs(expX-gotX) > diff {
			t.Errorf("expected x%d = %.[4]*[2]f got %.[4]*[3]f", i, gotX, expX, decimalAccuracy)
		}
	}
}
