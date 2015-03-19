import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.LAPACK
import Numeric.LinearAlgebra.Data

data Layer = Layer {
    nNeurons   :: Int,
    activation :: Double -> Double,
    weights    :: Vector Double
}

linearLayer :: Int -> Layer
linearLayer n = Layer { nNeurons = n, activation = id }

type Network = [Layer]

_activate :: Vector Double -> Layer -> Vector Double
_activate x layer = x  -------- TODO

activate :: Vector Double -> Network -> Vector Double
activate = foldl _activate

main :: IO()
main = do
    let a = linearLayer 10
    let b = linearLayer 20
    let c = linearLayer 2
    let net = [a,b,c]
    let x = vector [1..10]
    let y = activate x net
    print y