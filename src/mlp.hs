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
linearLayer n = Layer {
    neurons   = n,
    ϕ         = id,
    ϕ'        = id,
    layerType = "linear"
}

hyperbolicLayer :: Int -> Layer
hyperbolicLayer n = Layer {
    ϕ         = tanh,
    ϕ'        = \x -> 1 - tanh² x,
    neurons   = n,
    layerType = "tanh"
} where tanh² = tanh . tanh

sigmoidLayer :: Int -> Layer
sigmoidLayer n = Layer {
    ϕ         = sigmoid,
    ϕ'        = \x -> sigmoid x * (1 - sigmoid x),
    neurons   = n,
    layerType = "sigmoid"
} where sigmoid x = 1 / (1 + exp (-x))

-- TODO: It would be nicer to enforce #layers == #weights+1 by zipping.
mlp :: [Layer] -> [WeightMatrix] -> Network
mlp ls ws = debug Network { layers = ls, weights = ws, netType = "mlp" }
    where text  = map (\x -> "Layer " ++ show (fst x) ++ ": " ++ show (snd x)) (zip [1..] ls)
          debug = trace $ "Creating MLP\n" ++ intercalate "\n" text ++ "\n"

-- Initializer

initVector :: MTGen -> Int -> IO (Vector)
initVector rng n = do
    rs <- randoms rng
    let v = fromList (take n rs)
    return v

initMatrix :: MTGen -> Int -> Int -> IO (Matrix)
initMatrix rng n m = do
    rs <- randoms rng :: IO [FpType]
    let v = (n><m) $ take (n*m) rs
    return v

initConnections :: MTGen -> (Layer, Layer) -> IO (WeightMatrix)
initConnections rng (l₁,l₂) = do
    w <- initMatrix rng (neurons l₂) (neurons l₁)
    return w

initMlp :: MTGen -> [Layer] -> IO (Network)
initMlp rng ls = do
    ws <- mapM (initConnections rng) $ zip ls (tail ls)
    let net = mlp ls ws
    return net

-- Helper

vecToStr :: Vector -> String
vecToStr v = "[ " ++ intercalate "  " (map show (toList v)) ++ " ]"

vecsToStr :: [Vector] -> String
vecsToStr vs = intercalate "\n" (map vecToStr vs)

-- Layers

data Layer = Layer {
    ϕ         :: FpType -> FpType,
    ϕ'        :: FpType -> FpType,
    neurons   :: Int,
    layerType :: String
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
forward x (l,w) = mapVector (ϕ l) $ w <> x

activate :: Vector -> Network -> [Vector]
activate x (Network { layers = ls, weights = ws }) = debug result
    where result          = scanl forward input $ zip (tail ls) ws
          input           = mapVector inputActivation x
          inputActivation = ϕ (head ls)
          debug           = trace ("activate input\n" ++ vecToStr x ++ "\n\nactivate output\n" ++ vecsToStr result ++ "\n")

-- Datasets

-- TODO: decouple classification/regression datasets properly, add bias
data Dataset = Dataset [(Vector, Target)]
data Target = Int | FpType deriving (Eq,Show)

instance Show Dataset where
    show (Dataset d) = "dataset: " ++ show l ++ " elements, " ++ show u ++ " targets\n"
        where u = nub (map snd d)
              l = length d

-- Testing

main :: IO()
main = do
    rng <- newMTGen Nothing
    let l₁ = linearLayer 2
    let l₂ = sigmoidLayer 3
    let l₃ = linearLayer 1
    let layers = [l₁,l₂,l₃]

    net <- initMlp rng layers

    x <- initVector rng 2
    putStrLn $ "x = " ++ vecToStr x ++ "\n"

    let y = activate x net
    putStrLn $ "y = " ++ vecToStr (last y)
