import pandas as pd


def head_csv(input_filename, output_filename, num_samples):
    dfall = pd.read_csv(input_filename)
    dfslice = dfall.head(num_samples)
    dfslice.to_csv(output_filename, index=False)

# function to get unique values 
def unique(list1):       
    list_set = set(list1) 
    unique_list = (list(list_set)) 
    unique_list.sort()
    return unique_list


# returns the list of unique classids from the last column
def create_userids( df ):
    array = df.values
    y = array[:, -1]
    return unique( y )


# create a dictionary containing the amount of data of each class
#  input: dataframe with a classid in the last column
#  output: dictionary containing the frequencies
# 
def create_userid_dictionary( df ):
    array = df.values
    y = array[:, -1]
    unique_list = (list(  set(y) ))
    # Creating an empty dictionary  
    freq = {} 
    my_list = list(y)
    for items in my_list: 
        freq[items] = my_list.count(items) 
    return freq

# returns the number of keys in a dictionary
def dictionary_numkeys( dict ):
    len(list(dict.keys())) 

# print a list
def print_list( l ):
    sl = l.sort()
    for x in l:
        print(x)


# encode userids (classids) to 0, 1, 2, ..
# input: dataframe
# output: dataframe with encoded classids
def encodeUserids( df ):
    num_samples, num_features = df.shape
    array = df.values
    print("userid:" + str(array[0,-1]))
    X = array[:, 0: num_features-1]
    print(X.shape)
    y = array[:, -1]
    unique_list = (list(  set(y) ))
    print("unique ids: "+ str( len(unique_list) )) 

    # Creating an empty dictionary  
    freq = {} 
    counter = 0
    for items in unique_list: 
        freq[items] = counter
        counter = counter + 1
    print(freq)
    print(y)
    for i in range(0, num_samples):
        y[i] = freq[y[i]]
    print(y)
    df = pd.DataFrame(X)
    df['user'] = y
    print("new userid: "+str(df.values[0,-1]))
    return df