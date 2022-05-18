"""
Name: movie_recommendations.py
Date: 3/22/2021
Author: Alex Bae, Galen Forbes-Roberts, Josue Bautista
Description: We created a program that predicts the rating that a user would give
             to a movie he has not seen yet. This is accomplished by using the 
             ratings the user had already given for movies he had already watched. 
             Predicted and actual ratings are compared to test the accuracy of the program.
"""
import math
import csv
from scipy.stats import pearsonr

class BadInputError(Exception):
    pass

class Movie_Recommendations:
    # Constructor
    def __init__(self, movie_filename, training_ratings_filename):
        """
        Initializes the Movie_Recommendations object from 
        the files containing movie names and training ratings.  
        The following instance variables should be initialized:
        self.movie_dict - A dictionary that maps a movie id to
               a movie objects (objects the class Movie)
        self.user_dict - A dictionary that maps user id's to a 
               a dictionary that maps a movie id to the rating
               that the user gave to the movie.    
        """ 
        #Compile movie_dict and user_dict
        self.movie_dict = self.makeMovieDict(movie_filename) 

        self.user_dict = self.makeUserDict(training_ratings_filename)
        
    def makeMovieDict(self,movie_filename):
        """
        Compile a dictionary that maps movie id to movie objects. 
        Create self.movie_dict.
        """
        self.movie_dict = {}
        mf = open(movie_filename, encoding = 'utf-8')
        mf.readline()
        csv_reader = csv.reader(mf, delimiter = ',', quotechar = '"')
        for line in csv_reader:
            self.movie_dict[int(line[0])] = Movie(line[0],line[1])  #key = movie id: value = object
        mf.close()
        return self.movie_dict

    def makeUserDict(self,training_ratings_filename):
        """
        Create a dictionary, self.user_dict, that maps user ids to 
        another dictionary that maps the movies each user has rated 
        to the ratings each user gave. We get data from training_ratings
        file.
        """
        self.user_dict = {}
        tr = open(training_ratings_filename, encoding = 'utf-8')
        tr.readline()
        csv_reader = csv.reader(tr, delimiter = ',', quotechar = '"')
        for line in csv_reader:
            userID = int(line[0]) #Userid
            movieID = int(line[1]) #MovieID
            rating = round(float(line[2]), 2) #rating for movie by user
            #Assign a new dictionary for each new user
            usr_rating = self.user_dict.setdefault(userID,{})
            usr_rating[movieID] = rating
            if userID not in self.movie_dict[movieID].users:
                self.movie_dict[movieID].users.append(userID)
        tr.close()
        return self.user_dict

    def predict_rating(self, user_id, movie_id):
        """
        Returns the predicted rating that user_id will give to the
        movie whose id is movie_id. 
        If user_id has already rated movie_id, return
        that rating.
        If either user_id or movie_id is not in the database,
        then BadInputError is raised.
        """
        if user_id not in self.user_dict or movie_id not in self.movie_dict:
            raise BadInputError
        if user_id in self.movie_dict[movie_id].users:
            return self.user_dict[user_id][movie_id]
        else:
            rxs_lst = [] #list for all rate*similarities
            sim_lst = [] #list for the similarities
            for movie in self.user_dict[user_id]:
                rate_x = self.user_dict[user_id][movie] 
                sim_x = self.movie_dict[movie].get_similarity(movie_id, self.movie_dict, self.user_dict)
                sim_lst.append(sim_x) #append similarities separately to take sum later
                ratexsim = rate_x * sim_x  #rate * similarity
                rxs_lst.append(ratexsim) 
            if sum(sim_lst) == 0:
                return 2.5 
            predicted_rating = sum(rxs_lst)/sum(sim_lst) #To find predicted rating...
            return predicted_rating
        
    def predict_ratings(self, test_ratings_filename):
        """
        Returns a list of tuples, one tuple for each rating in the
        test ratings file.
        The tuple should contain
        (user id, movie title, predicted rating, actual rating)
        """
        predicted_ratings = []
        trf = open(test_ratings_filename,'r')
        trf.readline()
        csv_reader = csv.reader(trf, delimiter = ',', quotechar = '"')
        for line in csv_reader:
            user_id = int(line[0])
            movie = str(self.movie_dict[int(line[1])].title)
            test_rating = round(float(line[2]), 1)
            predicted_rating = self.predict_rating(user_id,int(line[1])) #call predict_rating for each line
            tup = (user_id, movie, predicted_rating, test_rating)
            predicted_ratings.append(tup)
        trf.close()
        return predicted_ratings

    def correlation(self, predicted_ratings, actual_ratings):
        """
        Returns the correlation between the values in the list predicted_ratings
        and the list actual_ratings.  The lengths of predicted_ratings and
        actual_ratings must be the same.
        """
        return pearsonr(predicted_ratings, actual_ratings)[0]
        
class Movie: 
    """
    Represents a movie from the movie database.
    """
    def __init__(self, id, title):
        """ 
        Constructor.
        Initializes the following instances variables.  You
        must use exactly the same names for your instance 
        variables.  (For testing purposes.)
        id: the id of the movie
        title: the title of the movie
        users: list of the id's of the users who have
            rated this movie.  Initially, this is
            an empty list, but will be filled in
            as the training ratings file is read.
        similarities: a dictionary where the key is the
            id of another movie, and the value is the similarity
            between the "self" movie and the movie with that id.
            This dictionary is initially empty.  It is filled
            in "on demand", as the file containing test ratings
            is read, and ratings predictions are made.
        """   
        #Initialize global variables
        self.title = title
        self.id = int(id)
        self.users = []
        self.similarities = {}
     
    def __str__(self):
        """
        Returns string representation of the movie object.
        Handy for debugging.
        """
        return '({},{})'.format(self.id,self.title)

    def __repr__(self):
        """
        Returns string representation of the movie object.
        """
        return '({},{},{},{})'.format(self.id,self.title,self.users,self.similarities)

    def get_similarity(self, other_movie_id, movie_dict, user_dict):
        """ 
        Returns the similarity between the movie that 
        called the method (self), and another movie whose
        id is other_movie_id.  (Uses movie_dict and user_dict)
        If the similarity has already been computed, return it.
        If not, compute the similarity (using the compute_similarity
        method), and store it in both
        the "self" movie object, and the other_movie_id movie object.
        Then return that computed similarity.
        If other_movie_id is not valid, raise BadInputError exception.
        """       
        if other_movie_id not in movie_dict:
            raise BadInputError

        similarity = self.compute_similarity(other_movie_id, movie_dict, user_dict)
        if similarity in movie_dict[self.id].similarities and similarity in movie_dict[other_movie_id].similarities:
            return similarity
        else:
            movie_dict[self.id].similarities[other_movie_id] = similarity
            movie_dict[other_movie_id].similarities[self.id] = similarity
            return similarity

    def compute_similarity(self, other_movie_id, movie_dict, user_dict):
        """ 
        Computes and returns the similarity between the movie that 
        called the method (self), and another movie whose
        id is other_movie_id.  (Uses movie_dict and user_dict)
        """
        diff_lst = []
        #I made a list for the users that viewed BOTH movies
        usr_both = set(movie_dict[self.id].users).intersection(set(movie_dict[other_movie_id].users))
        if len(usr_both) == 0:
            return 0
        for x in usr_both:
            diff = user_dict[x].get(self.id) - user_dict[x].get(other_movie_id)
            diff_lst.append(abs(diff))
        diffs_avg = sum(diff_lst)/len(diff_lst)
        similarity = 1 - diffs_avg/4.5
        return similarity

if __name__ == "__main__":
    # Create movie recommendations object.
    movie_recs = Movie_Recommendations("movies.csv", "training_ratings.csv")
    # Predict ratings for user/movie combinations
    rating_predictions = movie_recs.predict_ratings("test_ratings.csv")
    print("Rating predictions: ")
    for prediction in rating_predictions:
        print(prediction)
    predicted = [rating[2] for rating in rating_predictions]
    actual = [rating[3] for rating in rating_predictions]
    correlation = movie_recs.correlation(predicted, actual)
    print(f"Correlation: {correlation}")    