package neldermead

import (
	"math"
	"sort"
)

type Simplex struct {
	Points []Point
}

type Point struct {
	X []float64
	F float64
}

type Options struct {
	Alpha, Beta, Gamma, Delta, Tolerance float64
	MaxIterations                        int
	Constraints                          [][]float64
}

func Run(f func(x []float64) float64, x0 []float64, options Options) (Point, error) {
	simplex, err := createSimplex(x0, len(x0), options.Constraints)
	if err != nil {
		return Point{}, err
	}

	for i := 0; i < len(simplex.Points); i++ {
		simplex.Points[i].F = f(simplex.Points[i].X)
	}

	sortSimplex(simplex)

	for iter := 0; iter < options.MaxIterations; iter++ {
		if math.Abs(simplex.Points[0].F-simplex.Points[len(simplex.Points)-1].F) < options.Tolerance {
			return simplex.Points[0], nil
		}
		centroid := computeCentroid(simplex, len(simplex.Points)-1)
		reflectedPoint := reflectPoint(simplex.Points[len(simplex.Points)-1], centroid, options.Alpha)
		if reflectedPoint.F < simplex.Points[len(simplex.Points)-2].F {
			expandedPoint := reflectPoint(reflectedPoint, centroid, options.Gamma)
			if expandedPoint.F < reflectedPoint.F {
				replacePoint(simplex, len(simplex.Points)-1, expandedPoint)
			} else {
				replacePoint(simplex, len(simplex.Points)-1, reflectedPoint)
			}
		} else {
			if reflectedPoint.F < simplex.Points[len(simplex.Points)-1].F {
				replacePoint(simplex, len(simplex.Points)-1, reflectedPoint)
			}
			contractedPoint := reflectPoint(simplex.Points[len(simplex.Points)-1], centroid, options.Beta)
			if contractedPoint.F < simplex.Points[len(simplex.Points)-1].F {
				replacePoint(simplex, len(simplex.Points)-1, contractedPoint)
			} else {
				shrinkSimplex(simplex, options.Delta)
			}
		}
		for i := 0; i < len(simplex.Points); i++ {
			simplex.Points[i].F = f(simplex.Points[i].X)
		}
		sortSimplex(simplex)
		if len(options.Constraints) > 0 {
			err := checkConstraints(simplex.Points[0].X, options.Constraints)
			if err != nil {
				return Point{}, err
			}
		}
	}

	return simplex.Points[0], nil
}

func createSimplex(x []float64, n int, constraints [][]float64) (Simplex, error) {
	simplex := Simplex{Points: make([]Point, n+1)}

	for i := range simplex.Points {
		simplex.Points[i].X = make([]float64, len(x))
	}

	for j := 0; j < n; j++ {
		simplex.Points[0].X[j] = x[j]
	}

	for i := 1; i <= n; i++ {
		for j := 0; j < n; j++ {
			if i-1 == j {
				simplex.Points[i].X[j] = x[j] + 1.0
			} else {
				simplex.Points[i].X[j] = x[j]
			}
		}
	}

	if len(constraints) > 0 {
		for i := 0; i < len(simplex.Points); i++ {
			err := checkConstraints(simplex.Points[i].X, constraints)
			if err != nil {
				return Simplex{}, err
			}
		}
	}

	return simplex, nil
}

func sortSimplex(simplex Simplex) {
	sort.Slice(simplex.Points, func(i, j int) bool {
		return simplex.Points[i].F < simplex.Points[j].F
	})
}

func computeCentroid(simplex Simplex, excludeIndex int) []float64 {
	centroid := make([]float64, len(simplex.Points[0].X))
	for i := 0; i < len(simplex.Points); i++ {
		if i != excludeIndex {
			for j := 0; j < len(simplex.Points[i].X); j++ {
				centroid[j] += simplex.Points[i].X[j]
			}
		}
	}

	for j := 0; j < len(centroid); j++ {
		centroid[j] /= float64(len(simplex.Points) - 1)
	}

	return centroid
}

func shrinkSimplex(simplex Simplex, delta float64) {
	bestPoint := simplex.Points[0]
	for i := 1; i < len(simplex.Points); i++ {
		for j := 0; j < len(simplex.Points[i].X); j++ {
			simplex.Points[i].X[j] = bestPoint.X[j] + delta*(simplex.Points[i].X[j]-bestPoint.X[j])
		}
	}
}

func reflectPoint(p Point, centroid []float64, alpha float64) Point {
	reflectedPoint := Point{X: make([]float64, len(p.X))}
	for j := 0; j < len(p.X); j++ {
		reflectedPoint.X[j] = centroid[j] + alpha*(centroid[j]-p.X[j])
	}
	return reflectedPoint
}

func replacePoint(simplex Simplex, i int, newPoint Point) {
	simplex.Points[i] = newPoint
}

func checkConstraints(x []float64, constraints [][]float64) error {
	for _, c := range constraints {
		if len(c) != len(x) {
			return ErrorWrongDimension{}
		}
		for i := 0; i < len(x); i++ {
			if x[i] < c[i] {
				x[i] = c[i]
			}
		}
	}
	return nil
}

type ErrorWrongDimension struct{}

func (ErrorWrongDimension) Error() string {
	return "constraint has wrong number of dimensions"
}
