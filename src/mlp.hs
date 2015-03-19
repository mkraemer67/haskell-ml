import Debug.Trace
import Numeric.LinearAlgebra
import System.Random.Mersenne


-- Helpers

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


-- Layers

data Layer = Layer {
    activation :: Double -> Double,
    neurons    :: Int,
    layerType  :: String
}

instance Show Layer where
    show (Layer { neurons = n, layerType = k }) = k ++ ", " ++ show n ++ " neurons"

linearLayer :: Int -> Layer
linearLayer n = Layer { neurons = n, activation = id, layerType = "linear" }

hyperbolicLayer :: Int -> Layer
hyperbolicLayer n = Layer { neurons = n, activation = tanh, layerType = "tanh" }

sigmoidLayer :: Int -> Layer
sigmoidLayer n = Layer { neurons = n, activation = sigm, layerType = "sigmoid" }
    where sigm x = 1 / (1 + exp (-x))


-- Networks

data Network = Network {
    layers  :: [Layer],
    weights :: [Matrix Double],
    netType :: String
}

mlp :: [Layer] -> [Matrix Double] -> Network
mlp ls ws = Network { layers = ls, weights = ws, netType = "mlp" }

forward :: Vector Double -> (Layer, Matrix Double) -> Vector Double
forward x (l,w) = x ------------ TODO

activate :: Vector Double -> Network -> [Vector Double]
activate x (Network { layers = ls, weights = ws }) =
    trace ("activate\nin: " ++ show x ++ "\nout: " ++ show result ++ "\n") $ result
        where result          = scanl forward (mapVector inputActivation x) $ zip (drop 1 ls) ws
              inputActivation = activation (head ls)

initConnections :: MTGen -> (Layer, Layer) -> IO (Matrix Double)
initConnections g ((Layer { neurons = n }), (Layer { neurons = m })) = do
    w <- initMatrix g n m
    return w

initMlp :: MTGen -> [Layer] -> IO (Network)
initMlp g ls = do
    ws <- mapM (initConnections g) $ zip ls (tail ls)
    let net = mlp ls ws
    return net


-- Testing

main :: IO()
main = do
    g <- newMTGen Nothing
    let sizes = [2,3,1]
    let layers = map linearLayer sizes
    mapM_ (\x -> putStrLn ("Layer " ++ show (fst x) ++ ": " ++ show (snd x) ++ "")) (zip [1..] layers)
    net <- initMlp g layers

    x <- initVector g 2
    putStrLn $ "\nInput: " ++ show x ++ "\n"

    let y = activate x net
    putStrLn $ "Output: " ++ show y ++ "\n"