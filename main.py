import sys
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Sample user data
user_data = [
    ['712664', '5', '3'], ['1331154', '4', '3'], ['2632461', '3', '5'], ['44937', '5', '30'],
    ['656399', '4', '3'], ['439011', '1', '5'], ['1644750', '3', '3'], ['2031561', '4', '3'],
    ['616720', '4', '3'], ['2467008', '2', '4'], ['712664', '1', '43'], ['712664', '2', '35'],
    ['712664', '5', '34'], ['712664', '4', '33'], ['712664', '3', '1'], ['387952', '2', '10'],
    ['295630', '3', '8'], ['872561', '4', '12'], ['126789', '5', '20'], ['392014', '1', '25'],
    ['604821', '2', '15'], ['509243', '3', '18'], ['712456', '4', '22'], ['972431', '5', '28'],
    ['843210', '1', '31'], ['513209', '2', '38'], ['754320', '3', '40'], ['609827', '4', '36'],
    ['312589', '5', '45'], ['437820', '1', '48'], ['783405', '2', '42'], ['592837', '3', '39'],
    ['871203', '4', '32'], ['698743', '5', '29'], ['304985', '1', '27'], ['439085', '2', '24'],
    ['620934', '3', '21'], ['856412', '4', '17'], ['372948', '5', '14'], ['204875', '1', '11'],
    ['956783', '2', '9'], ['518934', '3', '6'], ['783049', '4', '4'], ['295840', '5', '2'],
    ['104592', '1', '7'], ['345928', '2', '13'], ['798403', '3', '16'], ['203985', '4', '19'],
    ['438905', '5', '23'], ['734859', '1', '26'], ['582490', '2', '33'], ['238957', '3', '37'],
    ['905627', '4', '41'], ['607982', '5', '44'], ['234509', '1', '46'], ['875420', '2', '47'],
    ['683459', '3', '49'], ['532489', '4', '50'], ['927483', '5', '51'], ['732984', '1', '52'],
    ['234859', '2', '53'], ['283495', '3', '54'], ['729384', '4', '55'], ['928345', '5', '56'],
    ['712664', '3', '21'], ['712664', '1', '51'], ['712664', '4', '50'], ['712664', '5', '52'],
    ['783049', '4', '53'], ['1000', '4', '12'], ['1000', '2', '34'], ['1000', '5', '47'], ['1001', '3', '15'], ['1001', '1', '26'],
    ['1001', '4', '37'], ['1002', '5', '3'], ['1002', '2', '18'], ['1002', '3', '29'], ['1002', '4', '42'],
    ['1003', '1', '7'], ['1003', '5', '22'], ['1003', '4', '41'], ['1004', '2', '19'], ['1004', '4', '33'],
    ['1004', '3', '49'], ['1005', '5', '10'], ['1005', '2', '25'], ['1005', '1', '36'], ['1006', '3', '5'],
    ['1006', '4', '11'], ['1006', '2', '21'], ['1007', '5', '14'], ['1007', '1', '31'], ['1007', '3', '40'],
    ['1008', '4', '8'], ['1008', '2', '30'], ['1008', '5', '50'], ['1009', '3', '6'], ['1009', '1', '17'],
    ['1009', '4', '23'], ['1010', '5', '9'], ['1010', '3', '12'], ['1010', '2', '27'], ['1011', '4', '1'],
    ['1011', '1', '13'], ['1011', '5', '35'], ['1012', '2', '2'], ['1012', '3', '28'], ['1012', '4', '45'],
    ['1013', '1', '4'], ['1013', '5', '20'], ['1013', '2', '38'], ['1014', '4', '16'], ['1014', '3', '24'],
    ['1014', '5', '32'], ['1015', '2', '7'], ['1015', '4', '39'], ['1015', '1', '48']
]

# Sample movie data
movie_data = [
    ['1', '1994', 'The Shawshank Redemption'], ['2', '1972', 'The Godfather'], ['3', '2008', 'The Dark Knight'],
    ['4', '1999', 'Fight Club'], ['5', '1994', 'Pulp Fiction'], ['6', '2003', 'The Lord of the Rings: The Return of the King'],
    ['7', '2010', 'Inception'], ['8', '1994', 'Forrest Gump'], ['9', '2001', 'The Lord of the Rings: The Fellowship of the Ring'],
    ['10', '2012', 'The Avengers'], ['11', '1999', 'The Matrix'], ['12', '1995', 'Se7en'],
    ['13', '2014', 'Interstellar'], ['14', '1995', 'Braveheart'], ['15', '1991', 'The Silence of the Lambs'],
    ['16', '1994', 'The Lion King'], ['17', '2001', 'Spirited Away'], ['18', '2000', 'Gladiator'],
    ['19', '2015', 'Mad Max: Fury Road'], ['20', '1998', 'Saving Private Ryan'], ['21', '2019', 'Joker'],
    ['22', '1999', 'American Beauty'], ['23', '2014', 'Guardians of the Galaxy'], ['24', '1980', 'The Shining'],
    ['25', '1985', 'Back to the Future'], ['26', '1984', 'Ghostbusters'], ['27', '2019', 'Parasite'],
    ['28', '2017', 'Dunkirk'], ['29', '2018', 'Black Panther'], ['30', '1977', 'Star Wars: Episode IV - A New Hope'],
    ['31', '2009', 'Avatar'], ['32', '2000', 'Memento'], ['33', '2014', 'The Grand Budapest Hotel'],
    ['34', '1999', 'The Sixth Sense'], ['35', '2004', 'Eternal Sunshine of the Spotless Mind'],
    ['36', '2010', 'Toy Story 3'], ['37', '1993', "Schindler's List"], ['38', '2006', 'The Departed'],
    ['39', '2012', 'Skyfall'], ['40', '2011', 'Drive'], ['41', '2010', 'The Social Network'],
    ['42', '2007', 'No Country for Old Men'], ['43', '2001', 'Shrek'], ['44', '1997', 'Titanic'],
    ['45', '1981', 'Raiders of the Lost Ark'], ['46', '1999', 'The Green Mile'], ['47', '2013', 'Frozen'],
    ['48', '2002', 'The Bourne Identity'], ['49', '2006', 'Casino Royale'], ['50', '2016', 'La La Land'],
    ['51', '1995', 'Heat'], ['52', '2008', 'WALL-E'], ['53', '2010', 'How to Train Your Dragon'],
    ['54', '1986', 'Top Gun'], ['55', '1991', 'Terminator 2: Judgment Day'], ['56', '1995', 'Toy Story'],
    ['57', '2009', 'Up'], ['58', '2018', 'Spider-Man: Into the Spider-Verse'], ['59', '1992', 'Reservoir Dogs']
]

# Convert movie data to dictionaries for easy lookup
movie_id_to_name = {movie[0]: movie[2] for movie in movie_data}
movie_name_to_id = {movie[2].lower(): movie[0] for movie in movie_data}

# Function to separate user ratings from user_data
def separate_user_ratings(data_list):
    user_ratings_dict = {}
    for row in data_list:
        user_id, rating, movie_id = row
        if user_id in user_ratings_dict:
            user_ratings_dict[user_id][movie_id] = int(rating)
        else:
            user_ratings_dict[user_id] = {movie_id: int(rating)}
    return user_ratings_dict

# Function to calculate cosine similarity between two users
def calculate_cosine_similarity(user1_ratings, user2_ratings):
    # Find common movies
    common_movies = set(user1_ratings.keys()) & set(user2_ratings.keys())
    if len(common_movies) == 0:
        return 0
    # Create rating vectors
    user1_vector = np.array([user1_ratings[movie] for movie in common_movies])
    user2_vector = np.array([user2_ratings[movie] for movie in common_movies])
    # Calculate cosine similarity
    return cosine_similarity([user1_vector], [user2_vector])[0][0]

# Function to recommend movies based on similar users
def recommend_movies(user_ratings_dict, current_user_id):
    similarities = []
    current_user_ratings = user_ratings_dict[current_user_id]
    for user_id, ratings in user_ratings_dict.items():
        if user_id != current_user_id:
            sim = calculate_cosine_similarity(current_user_ratings, ratings)
            similarities.append((user_id, sim))
    # Sort users by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    # Filter out users with zero similarity
    similarities = [user for user in similarities if user[1] > 0]
    if not similarities:
        return []
    # Get top similar users (e.g., top 3)
    top_users = [user_id for user_id, sim in similarities[:3]]
    # Aggregate recommendations from top similar users
    recommendations = {}
    for similar_user_id in top_users:
        similar_user_ratings = user_ratings_dict[similar_user_id]
        for movie_id, rating in similar_user_ratings.items():
            if movie_id not in current_user_ratings and rating >= 4:
                if movie_id in recommendations:
                    recommendations[movie_id].append(rating)
                else:
                    recommendations[movie_id] = [rating]
    # Calculate average rating for each recommended movie
    recommendations = [(movie_id, np.mean(ratings)) for movie_id, ratings in recommendations.items()]
    # Sort recommendations by average rating
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations

def main():
    print("Welcome to the Movie Recommender!")
    print("Please rate movies to get recommendations.")
    user_id = 'current_user'  # Placeholder user ID
    user_ratings = {}

    # Display movies for the user to rate
    print("\nList of movies:")
    for idx, movie in enumerate(movie_data, 1):
        print(f"{idx}. {movie[2]} ({movie[1]})")
    print("\nPlease rate at least one movie (ratings from 1 to 5). Type 'done' when finished.")

    while True:
        movie_input = input("\nEnter the movie number to rate (or 'done' to finish): ").strip()
        if movie_input.lower() == 'done':
            break
        if not movie_input.isdigit() or not (1 <= int(movie_input) <= len(movie_data)):
            print("Invalid input. Please enter a valid movie number.")
            continue
        movie_index = int(movie_input) - 1
        movie_id = movie_data[movie_index][0]
        movie_name = movie_data[movie_index][2]

        rating_input = input(f"Enter your rating for '{movie_name}' (1-5): ").strip()
        if not rating_input.isdigit() or not (1 <= int(rating_input) <= 5):
            print("Invalid rating. Please enter a number between 1 and 5.")
            continue
        rating = int(rating_input)
        user_ratings[movie_id] = rating
        print(f"You rated '{movie_name}' with a {rating}/5.")

    if not user_ratings:
        print("You did not rate any movies. Exiting the program.")
        sys.exit()

    # Update user_data with current user ratings
    global user_data
    for movie_id, rating in user_ratings.items():
        user_data.append([user_id, str(rating), movie_id])

    # Separate user ratings
    ratings_by_user = separate_user_ratings(user_data)

    # Get recommendations
    recommendations = recommend_movies(ratings_by_user, user_id)

    # Display recommendations
    print("\nRecommendations for you:")
    if not recommendations:
        print("No recommendations found. Try rating more movies.")
    else:
        for movie_id, rating in recommendations:
            movie_name = movie_id_to_name.get(movie_id, "Unknown Movie")
            print(f"{movie_name} (Predicted Rating: {rating:.1f}/5)")

    # Remove current user ratings from user_data to prevent duplication
    user_data = [row for row in user_data if row[0] != user_id]

if __name__ == '__main__':
    main()

