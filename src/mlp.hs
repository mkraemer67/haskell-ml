import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.LAPACK
import Numeric.LinearAlgebra.Data
import System.Random.Mersenne

data Layer = Layer {
    activation :: Double -> Double,
    weights    :: Vector Double,
    kind       :: String
}

linearLayer :: Vector Double -> Layer
linearLayer w = Layer { weights = w, activation = id, kind = "linear" }

hyperbolicLayer :: Vector Double -> Layer
hyperbolicLayer w = Layer { weights = w, activation = tanh, kind = "tanh" }

sigmoidLayer :: Vector Double -> Layer
sigmoidLayer w = Layer { weights = w, activation = sigm, kind = "sigmoid" }
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

instance Show Layer where
    show (Layer {activation = a, weights = w, kind = k}) = k ++ " " ++ show w

main :: IO()
main = do
    g <- newMTGen Nothing
    let sizes = [10,20,2]
    weights <- mapM (initVector g) sizes
    let net = map linearLayer weights
    mapM_ (\x -> putStrLn ("Layer " ++ show (fst x) ++ ": " ++ show (snd x))) (zip [1..] net)
    x <- initVector g 10
    putStrLn $ "Input: " ++ show x
    let y = activate x net
    putStrLn $ "Output: " ++ show y