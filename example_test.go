package neldermead_test

import (
	"fmt"
	"math"

	"github.com/crhntr/neldermead"
)

func ExampleRun() {
	// Define the objective function to be minimized.
	objective := func(x []float64) float64 {
		// y = x^3
		return math.Pow(x[0], 3) - x[1]
	}

	// Define the starting point for the optimizer.
	x0 := []float64{4, 5}

	// Define the optimization options.
	options := neldermead.NewOptions()
	options.Constraints = []neldermead.Constraint{
		{Min: 3, Max: 5},
		{Min: 0, Max: 10},
	}

	// Run the optimizer.
	result, err := neldermead.Run(objective, x0, options)
	if err != nil {
		panic(err)
	}

	// Print the result.
	fmt.Printf("Found minimum at x = %v, f(x) = %.2f\n", result.X, result.F)
	// Output: Found minimum at x = [3 10], f(x) = 17.00
}
