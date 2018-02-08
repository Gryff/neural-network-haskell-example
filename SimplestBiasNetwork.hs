import Control.Monad.Writer

alpha = 0.1
type Weight = Double
type Bias = Double
type Vector = [Double]
type TrainingSample = (Double, Double, Double)

cost :: Vector -> Vector -> Double
cost actual expected = (sum errors) / (fromIntegral $ 2 * length expected)
  where errors = [(yhat - y) ^ 2 | (yhat, y) <- zip actual expected]

errorDerivative :: (TrainingSample -> Double) -> Vector -> Vector -> Vector -> Double
errorDerivative withRespectTo actual expected input = (sum errors) / (fromIntegral $ length expected)
  where errors = map withRespectTo $ zip3 actual expected input

withRespectToM :: TrainingSample -> Double
withRespectToM (yhat, y, x) = (yhat - y) * x

withRespectToC :: TrainingSample -> Double
withRespectToC (yhat, y, _) = yhat - y

forward :: Vector -> Weight -> Bias -> Vector
forward input weight bias = map (\x -> x * weight + bias) input

backPropagate :: Double -> Double -> Double
backPropagate parameter errorDeriv = parameter - (alpha * errorDeriv)

neuralNet :: Weight -> Bias -> Vector -> Vector -> Writer [String] (Weight, Bias)
neuralNet weight bias input expectedOutput
  | closeEnough error = do
      tell ["Neural net found the answer!"]
      return (weight, bias)
  | otherwise = do
      tell ["Error is " ++ show error ++ ". Doing another round"]
      neuralNet newWeight newBias input expectedOutput
  where
    output = forward input weight bias
    error = cost output expectedOutput
    errorDerivWrtM = errorDerivative withRespectToM output expectedOutput input
    errorDerivWrtC = errorDerivative withRespectToC output expectedOutput input
    newWeight = backPropagate weight errorDerivWrtM
    newBias = backPropagate bias errorDerivWrtM

closeEnough :: Double -> Bool
closeEnough error = abs error < 0.0001

main :: IO ()
main = do
  let input = [1, 2, 3]
  let expectedOutput = [3.5, 5.5, 7.5]
  let initialWeight = 0.66
  let initialBias = 0.13456

  let ((finalWeight, finalBias), log) = runWriter $ neuralNet initialWeight initialBias input expectedOutput
  mapM_ putStrLn log
  putStrLn $ "final weight: " ++ show finalWeight
  putStrLn $ "final bias: " ++ show finalBias

