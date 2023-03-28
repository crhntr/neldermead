# A Nelder-Mead implementation in Go [![Go Reference](https://pkg.go.dev/badge/github.com/crhntr/neldermead.svg)](https://pkg.go.dev/github.com/crhntr/neldermead)

Nelder-Mead Optimization Algorithm in Go

This repository contains a Go implementation of the Nelder-Mead optimization algorithm, which is a gradient-free
optimization method for continuous, possibly non-convex, and noisy functions in low to moderate dimensions.

The algorithm uses a simplex, a polytope with n+1 vertices in n-dimensional space, to explore the search space. The
algorithm iteratively updates the simplex by reflecting, expanding, contracting, or shrinking it, based on the values of
the objective function at its vertices. The algorithm terminates when the difference in the objective function values
between the best and worst points in the simplex falls below the specified tolerance or when the maximum number of
iterations is reached.

## Installation

This package can be installed using Go modules with the following command:

```sh
go get github.com/crhntr/neldermead
```

## Usage

[Try it in the Go playground](https://go.dev/play/p/iLh2VgurPnf)

The Rosenbrock function is a well-known test function for optimization algorithms, and its global minimum is at (1, 1),
where the function value is 0. The Run function takes the Rosenbrock function as the objective function to minimize, the
initial guess x0 as the starting point, and the Options struct with the desired parameters for the Nelder-Mead algorithm.

The algorithm found a solution close to the global minimum with a very small objective function value.

The output of this program should be:

```
Solution: [1.0000101959363473 1.0000203935669024]
Objective function value: 4.421146657321306e-08
```

```go
package main

import (
	"fmt"
	"math"
	
	"github.com/crhntr/neldermead"
)

func main() {
	rosenbrock := func(x []float64) float64 {
		return math.Pow(1-x[0], 2) + 100*math.Pow(x[1]-x[0]*x[0], 2)
	}
	x0 := []float64{-1.2, 1.0} // initial guess
	options := neldermead.NewOptions()

	point, err := neldermead.Run(rosenbrock, x0, options)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	}
	fmt.Printf("Solution: %v\n", point.X)
	fmt.Printf("Objective function value: %v\n", point.F)
}
```
