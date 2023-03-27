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
		return math.Pow(x[0], 3)
	}

	// Define the starting point for the optimizer.
	x0 := []float64{4}

	// Define the optimization options.
	options := neldermead.NewOptionsWithConstraints([]neldermead.Constraint{
		{Min: 3, Max: 5},
	})

	// Run the optimizer.
	result, err := neldermead.Run(objective, x0, options)
	if err != nil {
		panic(err)
	}

	// Print the result.
	fmt.Printf("Found minimum at x = %.2f, f(x) = %.2f\n", result.X, result.F)
	// Output: Found minimum at x = [3.00], f(x) = 27.00
}
