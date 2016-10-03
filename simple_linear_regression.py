import graphlab


def simple_linear_regression(input_feature, output):
    # compute the sum of input_feature and output
    sum_input = input_feature.sum()
    sum_output = output.sum()
    num_data = output.size()

    # compute the product of the output and the input_feature and its sum
    product_of_inp_out = input_feature * output
    sum_product = product_of_inp_out.sum()
    # compute the squared value of the input_feature and its sum
    input_squared = input_feature*input_feature
    squared_sum  = input_squared.sum()
    # use the formula for the slope
    numerator = sum_product - ((float(1)/num_data) * (sum_input*sum_output))
    denominator = squared_sum - ((float(1)/num_data) * (sum_input*sum_input))
    slope = numerator/denominator
    # use the formula for the intercept
    intercept = output.mean()-(slope*input_feature.mean())

    return (intercept, slope)


def get_regression_predictions(input_feature, intercept, slope):
    # calculate the predicted values:
    predicted_values= slope*input_feature + intercept
    return predicted_values

def get_residual_sum_of_squares(input_feature, output, intercept, slope):
    # First get the predictions
    predicted_price = get_regression_predictions(input_feature, intercept,slope)
    # then compute the residuals (since we are squaring it doesn't matter which order you subtract)
    residual = output - (slope*input_feature+intercept)
    # square the residuals and add them up
    RSS = residual * residual
    return(RSS)

def inverse_regression_predictions(output, intercept, slope):
    # solve output = intercept + slope*input_feature for input_feature. Use this equation to compute the inverse predictions:
    estimated_feature = (output-intercept)/slope
    return estimated_feature

sales = graphlab.SFrame('kc_house_data.gl/')
print sales.head()
train_data,test_data = sales.random_split(.8,seed=0)
sqft_intercept, sqft_slope = simple_linear_regression(train_data['sqft_living'],
    train_data['price'])
print "Intercept: " + str(sqft_intercept)
print "Slope: " + str(sqft_slope)
