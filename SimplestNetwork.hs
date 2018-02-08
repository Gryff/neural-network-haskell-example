import Control.Monad.Writer

alpha = 0.1
type Weight = Double
type Vector = [Double]

cost :: Vector -> Vector -> Double
cost actual expected = (sum errors) / (fromIntegral $ 2 * length expected)
  where errors = [(yhat - y) ^ 2 | (yhat, y) <- zip actual expected]

errorDerivative :: Vector -> Vector -> Vector -> Double
errorDerivative actual expected input = (sum errors) / (fromIntegral $ length expected)
  where errors = [(yhat - y) * x | (yhat, y, x) <- zip3 actual expected input]

forward :: Vector -> Double -> Vector
forward input weight = map (* weight) input

backPropagate :: Double -> Double -> Double
backPropagate weight errorDeriv = weight - (alpha * errorDeriv)

neuralNet :: Weight -> Vector -> Vector -> Writer [String] Weight
neuralNet weight input expectedOutput
  | closeEnough error = do
      tell ["Neural net found the answer!"]
      return weight
  | otherwise = do
      tell ["Error is " ++ show error ++ ". Doing another round"]
      neuralNet newWeight input expectedOutput
  where
    output = forward input weight
    error = cost output expectedOutput
    errorDeriv = errorDerivative output expectedOutput input
    newWeight = backPropagate weight errorDeriv

closeEnough :: Double -> Bool
closeEnough error = abs error < 0.0001

main :: IO ()
main = do
  let input = [1, 2, 3]
  let expectedOutput = [2, 4, 6]
  let initialWeight = 0.66

  let (finalWeight, log) = runWriter $ neuralNet initialWeight input expectedOutput
  mapM_ putStrLn log
  putStrLn $ "finalWeight: " ++ show finalWeight

