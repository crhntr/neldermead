package neldermead

import (
	"errors"
	"math"
	"math/rand"
	"sort"
)

const (
	DefaultAlpha                       = 1.0
	DefaultBeta                        = 0.5
	DefaultGamma                       = 2.0
	DefaultDelta                       = 0.5
	DefaultTolerance                   = 0.01
	DefaultMaxIterations               = 1000
	OptionalSuggestedCollapseThreshold = 1e-8
	defaultToleranceAdjustment         = 1e-4
)

type Simplex struct {
	Points []Point
}

func (s *Simplex) replacePoint(i int, newPoint Point) {
	s.Points[i] = newPoint
}

type Point struct {
	X []float64
	F float64
}

type Function = func(x []float64) float64

type Options struct {
	Alpha,
	Beta,
	Gamma,
	Delta,
	Tolerance,
	CollapseThreshold float64

	MaxIterations int

	Constraints []Constraint
}

func NewDefaultOption() Options {
	return Options{
		Alpha:         DefaultAlpha,
		Beta:          DefaultBeta,
		Gamma:         DefaultGamma,
		Delta:         DefaultDelta,
		Tolerance:     DefaultTolerance,
		MaxIterations: DefaultMaxIterations,
	}
}

func NewOptionsWithSampling(f Function, constraints []Constraint, numSamples int) Options {
	o := NewDefaultOption()
	o.Alpha = alphaFromSampling(f, constraints, numSamples)
	o.Tolerance = calculateToleranceWithSampling(f, constraints, numSamples)
	o.Constraints = constraints
	return o
}

func NewOptionsWithConstraints(constraints []Constraint) Options {
	o := NewDefaultOption()
	o.Alpha = alphaFromConstraints(constraints)
	o.Tolerance = toleranceFromConstraints(constraints)
	o.Constraints = constraints
	return o
}

func calculateToleranceWithSampling(f Function, constraints []Constraint, numSamples int) float64 {
	if len(constraints) == 0 || numSamples <= 0 {
		return DefaultTolerance
	}
	functionValues := make([]float64, numSamples)
	for i := 0; i < numSamples; i++ {
		x := randomPointWithinConstraints(constraints)
		functionValues[i] = f(x)
	}
	minVal, maxVal := minMax(functionValues)
	estimatedRange := maxVal - minVal
	return estimatedRange * defaultToleranceAdjustment
}

func randomPointWithinConstraints(constraints []Constraint) []float64 {
	x := make([]float64, len(constraints))
	for i, c := range constraints {
		x[i] = c.Min + rand.Float64()*(c.Max-c.Min)
	}
	return x
}

func minMax(values []float64) (float64, float64) {
	minVal := values[0]
	maxVal := values[0]

	for _, v := range values {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}

	return minVal, maxVal
}

func toleranceFromConstraints(constraints []Constraint) float64 {
	maxRange := 0.0
	for _, c := range constraints {
		rangeC := c.Max - c.Min
		if rangeC > maxRange {
			maxRange = rangeC
		}
	}
	return maxRange / DefaultMaxIterations
}

func (options *Options) validate(x0 []float64) error {
	if options.Alpha <= 0 {
		return errors.New("invalid Options parameter: Alpha must be greater than 0")
	}

	if options.Beta <= 0 || options.Beta >= 1 {
		return errors.New("invalid Options parameter: Beta must be in the range (0, 1)")
	}

	if options.Gamma <= 1 {
		return errors.New("invalid Options parameter: Gamma must be greater than 1")
	}

	if options.Delta <= 0 || options.Delta >= 1 {
		return errors.New("invalid Options parameter: Delta must be in the range (0, 1)")
	}

	if options.Tolerance <= 0 {
		return errors.New("invalid Options parameter: Tolerance must be greater than 0")
	}

	if options.MaxIterations <= 0 {
		return errors.New("invalid Options parameter: MaxIterations must be greater than 0")
	}

	if len(options.Constraints) != 0 && len(options.Constraints) != len(x0) {
		return errors.New("invalid options: The number of constraints must match the length of x0")
	}

	for _, constraint := range options.Constraints {
		err := constraint.validate()
		if err != nil {
			return err
		}
	}

	if len(options.Constraints) != 0 {
		for i, x := range x0 {
			if x < options.Constraints[i].Min || x > options.Constraints[i].Max {
				return errors.New("invalid initial x parameter: x0 must satisfy the constraints")
			}
		}
	}

	return nil
}

type Constraint struct {
	Min, Max float64
}

func (c *Constraint) validate() error {
	if math.IsNaN(c.Min) || math.IsNaN(c.Max) {
		return errors.New("fields Min and Max must be valid numbers")
	}

	if math.IsInf(c.Min, 0) || math.IsInf(c.Max, 0) {
		return errors.New("fields Min and Max must not be infinite")
	}

	if c.Min > c.Max {
		return errors.New("fields Min must be less than or equal to Max")
	}

	return nil
}

func Run(f Function, x0 []float64, options Options) (Point, error) {
	err := options.validate(x0)
	if err != nil {
		return Point{}, err
	}

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
		reflectedPoint := simplex.Points[len(simplex.Points)-1].reflect(f, centroid, options.Alpha)
		if reflectedPoint.F < simplex.Points[len(simplex.Points)-2].F {
			expandedPoint := reflectedPoint.reflect(f, centroid, options.Gamma)
			if expandedPoint.F < reflectedPoint.F {
				simplex.replacePoint(len(simplex.Points)-1, expandedPoint)
			} else {
				simplex.replacePoint(len(simplex.Points)-1, reflectedPoint)
			}
		} else {
			if reflectedPoint.F < simplex.Points[len(simplex.Points)-1].F {
				simplex.replacePoint(len(simplex.Points)-1, reflectedPoint)
			}
			contractedPoint := simplex.Points[len(simplex.Points)-1].reflect(f, centroid, options.Beta)
			if contractedPoint.F < simplex.Points[len(simplex.Points)-1].F {
				simplex.replacePoint(len(simplex.Points)-1, contractedPoint)
			} else {
				shrinkSimplex(simplex, options.Delta)
			}
		}
		for i := 0; i < len(simplex.Points); i++ {
			simplex.Points[i].F = f(simplex.Points[i].X)
		}
		sortSimplex(simplex)
		if len(options.Constraints) > 0 {
			ensureXAreInConstraintBounds(simplex.Points[0].X, options.Constraints)
		}

		if options.CollapseThreshold != 0 {
			avgEdgeLength := simplex.averageEdgeLength()
			if avgEdgeLength < options.CollapseThreshold {
				return Point{}, errors.New("Simplex has collapsed")
			}
		}
	}

	return simplex.Points[0], nil
}

func createSimplex(x []float64, n int, constraints []Constraint) (Simplex, error) {
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
			ensureXAreInConstraintBounds(simplex.Points[i].X, constraints)
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

func (p *Point) reflect(f Function, centroid []float64, alpha float64) Point {
	reflectedPoint := Point{X: make([]float64, len(p.X))}
	for j := 0; j < len(p.X); j++ {
		reflectedPoint.X[j] = centroid[j] + alpha*(centroid[j]-p.X[j])
	}
	reflectedPoint.F = f(reflectedPoint.X)
	return reflectedPoint
}

func ensureXAreInConstraintBounds(x []float64, constraints []Constraint) {
	for i := range x {
		if x[i] < constraints[i].Min {
			x[i] = constraints[i].Min
		}
		if x[i] > constraints[i].Max {
			x[i] = constraints[i].Max
		}
	}
}

type ErrorWrongDimension struct{}

func (ErrorWrongDimension) Error() string {
	return "constraint has wrong number of dimensions"
}

func (s *Simplex) averageEdgeLength() float64 {
	n := len(s.Points)
	totalLength := 0.0
	count := 0

	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			dist := distance(s.Points[i].X, s.Points[j].X)
			totalLength += dist
			count++
		}
	}

	return totalLength / float64(count)
}

func distance(x1, x2 []float64) float64 {
	sum := 0.0
	for i := 0; i < len(x1); i++ {
		d := x1[i] - x2[i]
		sum += d * d
	}
	return math.Sqrt(sum)
}

func alphaFromSampling(f func(x []float64) float64, constraints []Constraint, numSamples int) float64 {
	if len(constraints) == 0 || numSamples <= 0 {
		return DefaultAlpha
	}
	functionValues := make([]float64, numSamples)

	for i := 0; i < numSamples; i++ {
		x := randomPointWithinConstraints(constraints)
		functionValues[i] = f(x)
	}

	minVal, maxVal := minMax(functionValues)
	estimatedRange := maxVal - minVal

	return estimatedRange * 0.01
}

func alphaFromConstraints(constraints []Constraint) float64 {
	if len(constraints) == 0 {
		return DefaultAlpha
	}
	maxRange := 0.0
	for _, c := range constraints {
		rangeC := c.Max - c.Min
		if rangeC > maxRange {
			maxRange = rangeC
		}
	}
	return maxRange * 0.01
}
