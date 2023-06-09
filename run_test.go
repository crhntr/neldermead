package neldermead

import (
	"math"
	"math/rand"
	"strings"
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

	t.Run("bad initial x", func(t *testing.T) {
		// Define the objective function to optimize
		objective := func(x []float64) float64 {
			return x[0] - x[1]
		}

		// Define the starting point and Constraints
		x := []float64{10, 10}
		constraints := []Constraint{
			{Min: -1, Max: 1},
			{Min: -1, Max: 1},
		}

		// Set the options for the optimizer
		options := NewOptions()
		options.Constraints = constraints

		// Run the optimizer
		_, err := Run(objective, x, options)
		if err == nil {
			t.Errorf("expected error not nil")
		}
	})
}

func TestSimplexCollapse(t *testing.T) {
	src := rand.New(rand.NewSource(101))
	flatRegionFunctionWithNoise := func(x []float64) float64 {
		sum := 0.0
		for _, xi := range x {
			noise := src.Float64() * 1e-10
			sum += ((xi - 5) * (xi - 5) * (xi - 5) * (xi - 5)) + noise
		}
		return sum
	}

	initialGuess := []float64{5.0, 5.0}
	options := Options{
		Alpha:             1.0,
		Beta:              0.5,
		Gamma:             2.0,
		Delta:             0.5,
		Tolerance:         1e-16,
		MaxIterations:     1000,
		CollapseThreshold: 1e-5,
	}

	_, err := Run(flatRegionFunctionWithNoise, initialGuess, options)
	if err == nil {
		t.Fatalf("expected failure")
	}
	if !strings.Contains(err.Error(), ErrorSimplexCollapse{}.Error()) {
		t.Errorf("expected error %q", ErrorSimplexCollapse{}.Error())
	}
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

func TestOptions_Validate(t *testing.T) {
	tests := []struct {
		name      string
		options   Options
		wantError bool
	}{
		{
			name: "Valid options",
			options: Options{
				Alpha:             DefaultAlpha,
				Beta:              DefaultBeta,
				Gamma:             DefaultGamma,
				Delta:             DefaultDelta,
				Tolerance:         DefaultTolerance,
				MaxIterations:     DefaultMaxIterations,
				CollapseThreshold: 1e-6,
				Constraints: []Constraint{
					{Min: -1, Max: 1},
				},
			},
			wantError: false,
		},
		{
			name: "Negative Alpha",
			options: Options{
				Alpha:         -1.0,
				Beta:          DefaultBeta,
				Gamma:         DefaultGamma,
				Delta:         DefaultDelta,
				Tolerance:     DefaultTolerance,
				MaxIterations: DefaultMaxIterations,
			},
			wantError: true,
		},
		{
			name: "Negative Beta",
			options: Options{
				Alpha:         DefaultAlpha,
				Beta:          -1,
				Gamma:         DefaultGamma,
				Delta:         DefaultDelta,
				Tolerance:     DefaultTolerance,
				MaxIterations: DefaultMaxIterations,
			},
			wantError: true,
		},
		{
			name: "Beta too large",
			options: Options{
				Alpha:         DefaultAlpha,
				Beta:          1.00001,
				Gamma:         DefaultGamma,
				Delta:         DefaultDelta,
				Tolerance:     DefaultTolerance,
				MaxIterations: DefaultMaxIterations,
			},
			wantError: true,
		},
		{
			name: "Gamma too small",
			options: Options{
				Alpha:         DefaultAlpha,
				Beta:          DefaultBeta,
				Gamma:         .999,
				Delta:         DefaultDelta,
				Tolerance:     DefaultTolerance,
				MaxIterations: DefaultMaxIterations,
			},
			wantError: true,
		},
		{
			name: "Delta too small",
			options: Options{
				Alpha:         DefaultAlpha,
				Beta:          DefaultBeta,
				Gamma:         DefaultGamma,
				Delta:         -1,
				Tolerance:     DefaultTolerance,
				MaxIterations: DefaultMaxIterations,
			},
			wantError: true,
		},
		{
			name: "Delta too large",
			options: Options{
				Alpha:         DefaultAlpha,
				Beta:          DefaultBeta,
				Gamma:         DefaultGamma,
				Delta:         1.1,
				Tolerance:     DefaultTolerance,
				MaxIterations: DefaultMaxIterations,
			},
			wantError: true,
		},
		{
			name: "Negative Tolerance",
			options: Options{
				Alpha:         DefaultAlpha,
				Beta:          DefaultBeta,
				Gamma:         DefaultGamma,
				Delta:         DefaultDelta,
				Tolerance:     -1,
				MaxIterations: DefaultMaxIterations,
			},
			wantError: true,
		},
		{
			name: "Zero MaxIterations",
			options: Options{
				Alpha:         DefaultAlpha,
				Beta:          DefaultBeta,
				Gamma:         DefaultGamma,
				Delta:         DefaultDelta,
				Tolerance:     DefaultTolerance,
				MaxIterations: 0,
			},
			wantError: true,
		},
		{
			name: "Negative MaxIterations",
			options: Options{
				Alpha:         DefaultAlpha,
				Beta:          DefaultBeta,
				Gamma:         DefaultGamma,
				Delta:         DefaultDelta,
				Tolerance:     DefaultTolerance,
				MaxIterations: -1,
			},
			wantError: true,
		},
		{
			name: "Constraint with equal upper and lower bounds",
			options: Options{
				Alpha:         DefaultAlpha,
				Beta:          DefaultBeta,
				Gamma:         DefaultGamma,
				Delta:         DefaultDelta,
				Tolerance:     DefaultTolerance,
				MaxIterations: DefaultMaxIterations,
				Constraints: []Constraint{
					{Min: 1, Max: 1},
				},
			},
			wantError: true,
		},
		{
			name: "Constraint values must not be infinite",
			options: Options{
				Alpha:         DefaultAlpha,
				Beta:          DefaultBeta,
				Gamma:         DefaultGamma,
				Delta:         DefaultDelta,
				Tolerance:     DefaultTolerance,
				MaxIterations: DefaultMaxIterations,
				Constraints: []Constraint{
					{Min: math.Inf(-1), Max: math.Inf(1)},
				},
			},
			wantError: true,
		},
		{
			name: "Constraint Max value must be a number",
			options: Options{
				Alpha:         DefaultAlpha,
				Beta:          DefaultBeta,
				Gamma:         DefaultGamma,
				Delta:         DefaultDelta,
				Tolerance:     DefaultTolerance,
				MaxIterations: DefaultMaxIterations,
				Constraints: []Constraint{
					{Min: math.NaN(), Max: 2},
				},
			},
			wantError: true,
		},
		{
			name: "Constraint Max value must be a number",
			options: Options{
				Alpha:         DefaultAlpha,
				Beta:          DefaultBeta,
				Gamma:         DefaultGamma,
				Delta:         DefaultDelta,
				Tolerance:     DefaultTolerance,
				MaxIterations: DefaultMaxIterations,
				Constraints: []Constraint{
					{Min: 2, Max: math.NaN()},
				},
			},
			wantError: true,
		},
		{
			name: "Constraint Min must be below Max",
			options: Options{
				Alpha:         DefaultAlpha,
				Beta:          DefaultBeta,
				Gamma:         DefaultGamma,
				Delta:         DefaultDelta,
				Tolerance:     DefaultTolerance,
				MaxIterations: DefaultMaxIterations,
				Constraints: []Constraint{
					{Min: 10, Max: 5},
				},
			},
			wantError: true,
		},
		// Add more test cases here...
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.options.validate()
			if (err != nil) != tt.wantError {
				t.Errorf("Options.Validate() error = %v, wantError %v", err, tt.wantError)
			}
		})
	}
}

func TestOptions_validateX0(t *testing.T) {
	tests := []struct {
		name    string
		cs      []Constraint
		x0      []float64
		wantErr bool
	}{
		{
			name:    "empty",
			wantErr: false,
		},
		{
			name: "x is within constraint",
			cs: []Constraint{
				{Min: 0, Max: 2},
			},
			x0:      []float64{1},
			wantErr: false,
		},
		{
			name: "x is not within constraint",
			cs: []Constraint{
				{Min: 0, Max: 2},
			},
			x0:      []float64{200},
			wantErr: true,
		},
		{
			name:    "no constraints",
			cs:      nil,
			x0:      []float64{200, 200},
			wantErr: false,
		},
		{
			name: "exactly at lower bound range",
			cs: []Constraint{
				{Min: -2, Max: 2},
			},
			x0:      []float64{-2},
			wantErr: false,
		},
		{
			name: "exactly at upper bound range",
			cs: []Constraint{
				{Min: -2, Max: 2},
			},
			x0:      []float64{2},
			wantErr: false,
		},
		{
			name: "wrong number of constraints",
			cs: []Constraint{
				{Min: -2, Max: 2},
			},
			x0:      []float64{2, 4, 5},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			options := &Options{
				Constraints: tt.cs,
			}
			if err := options.validateX0(tt.x0); (err != nil) != tt.wantErr {
				t.Errorf("validateX0() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
