#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    factor = 0.1
    cleaned_data = []

    ### your code goes here

    from operator import itemgetter

    errors = [ abs(net_worths[ii] - predictions[ii]) for ii in range(len(net_worths))]
    cleaned_data = [ (ages[ii], net_worths[ii], errors[ii]) for ii in range(len(net_worths))]

    cleaned_data = sorted(cleaned_data, key = itemgetter(2))
    percent = len(cleaned_data) - int(len(cleaned_data)*factor)
    cleaned_data = cleaned_data[:percent]
    
    return cleaned_data

