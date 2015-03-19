import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.LAPACK

data Layer = Layer {
    nNeurons   :: Int,
    activation :: Double -> Double
}

linearLayer :: Int -> Layer
linearLayer n = Layer { nNeurons = n, activation = id }

type Network = [Layer]

main :: IO()
main = do
    let
        a = (1000><1000) $ replicate (1000*1000) 1.0
        b = (1000><1000) $ replicate (1000*1000) 1.0
    print $ (a `multiplyR` b) @@> (900,900)