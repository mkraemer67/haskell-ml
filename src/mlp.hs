import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.LAPACK
import Numeric.LinearAlgebra.Data
import System.Random.Mersenne

data Layer = Layer {
    activation :: Double -> Double,
    weights    :: Vector Double
}

linearLayer :: Vector Double -> Layer
linearLayer w = Layer { weights = w, activation = id }

hyperbolicLayer :: Vector Double -> Layer
hyperbolicLayer w = Layer { weights = w, activation = tanh }

sigmoidLayer :: Vector Double -> Layer
sigmoidLayer w = Layer { weights = w, activation = sigm }
    where sigm x = 1 / (1 + exp (-x))

type Network = [Layer]

_activate :: Vector Double -> Layer -> Vector Double
_activate x layer = x  -------- TODO

activate :: Vector Double -> Network -> Vector Double
activate = foldl _activate

initVector :: MTGen -> Int -> IO (Vector Double)
initVector g n = do
    rs <- randoms g
    let v = fromList (take n rs)
    return v

main :: IO()
main = do
    g <- newMTGen Nothing
    let a = linearLayer 10
    let b = linearLayer 20
    let c = linearLayer 2
    let net = [a,b,c]
    x <- initVector g 10
    putStrLn $ "Input: " ++ show x
    let y = activate x net
    putStrLn $ "Output: " ++ show y