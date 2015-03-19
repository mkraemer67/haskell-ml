{-# LANGUAGE BangPatterns #-}

import Data.List
import Debug.Trace
import Numeric.LinearAlgebra
import System.Random.Mersenne

-- Generators

linearLayer :: Int -> Layer
linearLayer n = Layer { neurons = n, activation = id, layerType = "linear" }

hyperbolicLayer :: Int -> Layer
hyperbolicLayer n = Layer { neurons = n, activation = tanh, layerType = "tanh" }

sigmoidLayer :: Int -> Layer
sigmoidLayer n = Layer { neurons = n, activation = sigm, layerType = "sigmoid" }
    where sigm x = 1 / (1 + exp (-x))

mlp :: [Layer] -> [Matrix Double] -> Network
mlp ls ws = debug Network { layers = ls, weights = ws, netType = "mlp" }
    where debug = trace $ "Creating MLP\n" ++ intercalate "\n" text ++ "\n"
          text  = map (\x -> "Layer " ++ show (fst x) ++ ": " ++ show (snd x)) (zip [1..] ls)

-- Initializer

initVector :: MTGen -> Int -> IO (Vector Double)
initVector g n = do
    rs <- randoms g
    let v = fromList (take n rs)
    return v

initMatrix :: MTGen -> Int -> Int -> IO (Matrix Double)
initMatrix g n m = do
    rs <- randoms g :: IO [Double]
    let v = (n><m) $ take (n*m) rs
    return v

initConnections :: MTGen -> (Layer, Layer) -> IO (Matrix Double)
initConnections g (l1,l2) = do
    w <- initMatrix g (neurons l2) (neurons l1)
    return w

initMlp :: MTGen -> [Layer] -> IO (Network)
initMlp g ls = do
    ws <- mapM (initConnections g) $ zip ls (tail ls)
    let net = mlp ls ws
    return net

-- Helper

vecToStr :: Vector Double -> String
vecToStr v = "[ " ++ intercalate "  " (map show (toList v)) ++ " ]"

vecsToStr :: [Vector Double] -> String
vecsToStr vs = intercalate "\n" (map vecToStr vs)

-- Layers

data Layer = Layer {
    activation :: Double -> Double,
    neurons    :: Int,
    layerType  :: String
}

instance Show Layer where
    show (Layer { neurons = n, layerType = t }) = t ++ ", " ++ show n ++ " neurons"

-- Networks

data Network = Network {
    layers  :: [Layer],
    weights :: [Matrix Double],
    netType :: String
}

forward :: Vector Double -> (Layer, Matrix Double) -> Vector Double
forward x (l,w) = w <> x

activate :: Vector Double -> Network -> [Vector Double]
activate x (Network { layers = ls, weights = ws }) =
     debug result
        where result          = scanl forward input $ zip (drop 1 ls) ws
              input           = mapVector inputActivation x
              inputActivation = activation (head ls)
              debug           = trace ("activate input\n" ++ vecToStr x ++ "\n\nactivate output\n" ++ vecsToStr result ++ "\n")

-- Testing

main :: IO()
main = do
    g <- newMTGen Nothing
    let sizes = [2,3,1]
    let layers = map linearLayer sizes

    net <- initMlp g layers

    x <- initVector g 2
    putStrLn $ "x = " ++ vecToStr x ++ "\n"

    let !y = activate x net
    putStrLn $ "y = " ++ vecToStr (last y)
