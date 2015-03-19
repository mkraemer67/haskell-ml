{-# LANGUAGE BangPatterns #-}

import Data.List
import Debug.Trace
import Numeric.LinearAlgebra hiding (Matrix, Vector)
import qualified Numeric.LinearAlgebra as LA
import System.Random.Mersenne

type FpType = Double
type Vector = LA.Vector FpType
type Matrix = LA.Matrix FpType

-- Generators

linearLayer :: Int -> Layer
linearLayer n = Layer { neurons = n, activation = id, layerType = "linear" }

hyperbolicLayer :: Int -> Layer
hyperbolicLayer n = Layer { neurons = n, activation = tanh, layerType = "tanh" }

sigmoidLayer :: Int -> Layer
sigmoidLayer n = Layer { neurons = n, activation = sigm, layerType = "sigmoid" }
    where sigm x = 1 / (1 + exp (-x))

-- TODO: It would be nicer to enforce #layers == #weights+1 by zipping.
mlp :: [Layer] -> [WeightMatrix] -> Network
mlp ls ws = debug Network { layers = ls, weights = ws, netType = "mlp" }
    where text  = map (\x -> "Layer " ++ show (fst x) ++ ": " ++ show (snd x)) (zip [1..] ls)
          debug = trace $ "Creating MLP\n" ++ intercalate "\n" text ++ "\n"

-- Initializer

initVector :: MTGen -> Int -> IO (Vector)
initVector g n = do
    rs <- randoms g
    let v = fromList (take n rs)
    return v

initMatrix :: MTGen -> Int -> Int -> IO (Matrix)
initMatrix g n m = do
    rs <- randoms g :: IO [FpType]
    let v = (n><m) $ take (n*m) rs
    return v

initConnections :: MTGen -> (Layer, Layer) -> IO (WeightMatrix)
initConnections g (l1,l2) = do
    w <- initMatrix g (neurons l2) (neurons l1)
    return w

initMlp :: MTGen -> [Layer] -> IO (Network)
initMlp g ls = do
    ws <- mapM (initConnections g) $ zip ls (tail ls)
    let net = mlp ls ws
    return net

-- Helper

vecToStr :: Vector -> String
vecToStr v = "[ " ++ intercalate "  " (map show (toList v)) ++ " ]"

vecsToStr :: [Vector] -> String
vecsToStr vs = intercalate "\n" (map vecToStr vs)

-- Layers

data Layer = Layer {
    activation :: FpType -> FpType,
    neurons    :: Int,
    layerType  :: String
}

instance Show Layer where
    show (Layer { neurons = n, layerType = t }) = t ++ ", " ++ show n ++ " neurons"

-- Networks

data Network = Network {
    layers  :: [Layer],
    weights :: [WeightMatrix],
    netType :: String
}

type WeightMatrix = Matrix

forward :: Vector -> (Layer, WeightMatrix) -> Vector
forward x (l,w) = mapVector (activation l) $ w <> x

activate :: Vector -> Network -> [Vector]
activate x (Network { layers = ls, weights = ws }) =
     debug result
        where result          = scanl forward input $ zip (drop 1 ls) ws
              input           = mapVector inputActivation x
              inputActivation = activation (head ls)
              debug           = trace ("activate input\n" ++ vecToStr x ++ "\n\nactivate output\n" ++ vecsToStr result ++ "\n")

-- Datasets

data Dataset = Dataset [(Vector, Target)]
data Target = Int | FpType deriving (Eq,Show)

instance Show Dataset where
    show (Dataset d) = "dataset: " ++ show l ++ " elements, " ++ show u ++ " targets\n"
        where u = nub (map snd d)
              l = length d

-- Testing

main :: IO()
main = do
    g <- newMTGen Nothing
    let sizes = [2,5,1]
    let l1 = linearLayer 2
    let l2 = sigmoidLayer 3
    let l3 = linearLayer 1
    let layers = [l1,l2,l3]

    net <- initMlp g layers

    x <- initVector g 2
    putStrLn $ "x = " ++ vecToStr x ++ "\n"

    let !y = activate x net
    putStrLn $ "y = " ++ vecToStr (last y)
