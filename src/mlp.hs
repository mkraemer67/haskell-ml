{-# LANGUAGE BangPatterns #-}

import Control.Monad.ST
import Data.Char
import Data.List
import Debug.Trace
import Numeric.LinearAlgebra hiding (Matrix, Vector)
import qualified Numeric.LinearAlgebra as LA
import System.Random.Mersenne
import System.Random.Shuffle

-- Aliases

type FpType = Double
type Vector = LA.Vector FpType
type Matrix = LA.Matrix FpType

type Delta = FpType
type LearningRate = FpType
type Ratio = FpType

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
    w <- initMatrix rng (nNeurons l₂) (nNeurons l₁)
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

datasetToStr :: Dataset -> String
datasetToStr d = "dataset: " ++ show l ++ " elements, " ++ show u ++ " targets\n"
    where u = nub (map snd d)
          l = length d

-- Layers

--TODO: allow 2d tensors for activation as well to facilitate softmax
data Layer = Layer {
    ϕ         :: FpType -> FpType,
    ϕ'        :: FpType -> FpType,
    nNeurons  :: Int,
    layerType :: String
}

instance Show Layer where
    show (Layer { nNeurons = n, layerType = t }) = t ++ ", " ++ show n ++ " neurons"

linearLayer :: Int -> Layer
linearLayer n = Layer {
    ϕ         = id,
    ϕ'        = id,
    nNeurons  = n,
    layerType = "linear"
}

hyperbolicLayer :: Int -> Layer
hyperbolicLayer n = Layer {
    ϕ         = tanh,
    ϕ'        = \x -> 1 - tanh² x,
    nNeurons  = n,
    layerType = "tanh"
} where tanh² = tanh . tanh

sigmoidLayer :: Int -> Layer
sigmoidLayer n = Layer {
    ϕ         = sigmoid,
    ϕ'        = \x -> sigmoid x * (1 - sigmoid x),
    nNeurons  = n,
    layerType = "sigmoid"
} where sigmoid x = 0.5 * (1 + tanh (0.5 * x))

-- Networks

data Network = Network {
    layers  :: [Layer],
    weights :: [WeightMatrix],
    netType :: String
}

type WeightMatrix = Matrix

--TODO: It would be nicer to enforce #layers == #weights+1 by zipping.
mlp :: [Layer] -> [WeightMatrix] -> Network
mlp ls ws = debug Network { layers = ls, weights = ws, netType = "mlp" }
    where text  = map (\x -> "Layer " ++ show (fst x) ++ ": " ++ show (snd x)) (zip [1..] ls)
          debug = trace $ "Creating MLP\n" ++ intercalate "\n" text ++ "\n"

forward :: Vector -> (Layer, WeightMatrix) -> Vector
forward x (l,w) = mapVector (ϕ l) $ w <> x

activate :: Network -> Vector -> [Vector]
activate (Network { layers = ls, weights = ws }) x = debug result
    where result = scanl forward x $ zip (tail ls) ws
          debug  = trace ("activate input\n" ++ vecToStr x ++ "\n\nactivate output\n" ++ vecsToStr result ++ "\n")

-- Datasets

--TODO: decouple classification/regression datasets properly, add bias s.t. it works flawlessly in all usages
type Dataset = [(Vector, Vector)]

--TODO: shuffle and split
split :: MTGen -> Dataset -> Ratio -> (Dataset, Dataset)
split rng d r = (d,d)

-- Training

--TODO: add momentum μ
data Trainer = Trainer {
    η         :: LearningRate,
    terminate :: TrainState -> Bool
}

data TrainState = TrainState {
    nEpochs :: Int,
    δs      :: [Delta]
} deriving (Show)

--TODO: more useful termination criteria
nEpochTrainer :: LearningRate -> Int -> Trainer
nEpochTrainer lr n = debug Trainer { η = lr, terminate = (> n) . nEpochs }
    where debug = trace ("Creating nEpochTrainer\n" ++ "η = " ++ show lr ++ "\nmaxEpochs = " ++ show n ++ "\n")

_epoch :: Delta -> Dataset -> Network -> Trainer -> (Network, Delta)
_epoch δs ds net t = runST $ do
    return (net,0.0)
    --mapM_ f ds where
    --    f d = do
    --        let y = activate net (fst d)
    --        let δ = sum . (**2) $ (snd d) - y
    --        -- TODO: backpropagate
    --        _epoch (δs + δ) ds net t

epoch :: Dataset -> Network -> Trainer -> (Network, Delta)
epoch = _epoch 0

_train :: TrainState -> Dataset -> Network -> Trainer -> IO (Network)
_train s d net t = do
    let (net',δ') = epoch d net t
    let s' = update s where update s = TrainState { nEpochs = 1 + (nEpochs s), δs = δs s ++ [δ'] }
    let statusMessage = "δ_" ++ show (nEpochs s') ++ " = " ++ show δ' ++ "\n" ++ show s' ++ "\n"

    d' <- shuffleM d
    if (terminate t) s'
        then trace ("Finished Training.\n") return net'
        else trace statusMessage $ _train s' d' net' t

--TODO: support testing set + optional validation set
train :: Dataset -> Network -> Trainer -> IO (Network)
train = _train s₀ where s₀ = TrainState { nEpochs = 0, δs = [] }

--TODO
crossvalidate :: Int -> Dataset -> Network -> Trainer -> IO (Network)
crossvalidate n = trace ("Cross-validation is not implemented!") train

-- Evaluation

--TODO

-- Testing

main :: IO()
main = do
    putStrLn ""
    rng <- newMTGen Nothing

    let l₁ = linearLayer 5
    let l₂ = sigmoidLayer 3
    let l₃ = linearLayer 5
    let layers = [l₁,l₂,l₃]

    !net <- initMlp rng layers
    xs <- mapM (initVector rng) (take 100 $ repeat 5)

    let ys = xs
    let dataset = zip xs ys
    let trainer = nEpochTrainer 0.1 5
    !trainedNet <- train dataset net trainer

    putStrLn $ "Done!\n"
