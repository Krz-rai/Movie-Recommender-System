# Movie Recommender System

This Python project is a simple movie recommender system that recommends movies based on cosine similarity between users' movie ratings. By comparing your movie ratings with those of similar users, the system predicts and recommends movies that you are likely to enjoy.

## Features
- Input movie ratings from users.
- Calculate similarity between users using cosine similarity.
- Recommend movies that similar users have highly rated but the current user has not yet seen.

## Prerequisites

The project requires Python 3.x and the following libraries:
- `numpy`: For handling numerical data and arrays.
- `scikit-learn`: For calculating cosine similarity between users.

You can install the dependencies using the following command:

```bash
pip install numpy scikit-learn
```

## How It Works

1. **User Data Input**: Users provide movie ratings (1 to 5) for a selection of movies.
2. **Cosine Similarity Calculation**: The system compares your movie ratings with other users’ ratings to find users with similar taste.
3. **Movie Recommendations**: Based on these similar users, the system recommends movies that they rated highly but you haven't rated yet.

## Code Breakdown

### 1. **`separate_user_ratings(data_list)`**

This function processes the `user_data` and organizes it into a dictionary where each key is a user ID, and each value is a dictionary of movie ratings by that user.

**Parameters:**
- `data_list`: A list containing user IDs, movie ratings, and movie IDs.

**Returns:**
- A dictionary where keys are user IDs and values are dictionaries of movie ratings.

### 2. **`calculate_cosine_similarity(user1_ratings, user2_ratings)`**

This function calculates the cosine similarity between two users based on their common movie ratings.

**Parameters:**
- `user1_ratings`: A dictionary of the first user's movie ratings.
- `user2_ratings`: A dictionary of the second user's movie ratings.

**Returns:**
- The cosine similarity between the two users. If there are no common movie ratings, it returns 0.

### 3. **`recommend_movies(user_ratings_dict, current_user_id)`**

This function provides movie recommendations for a user by finding the most similar users and recommending movies they liked but the current user hasn’t seen.

**Parameters:**
- `user_ratings_dict`: A dictionary of user ratings.
- `current_user_id`: The user ID of the current user.

**Returns:**
- A list of recommended movies with predicted ratings, sorted by the predicted rating.

### 4. **`main()`**

The main function handles user interaction, including prompting the user to rate movies, calculating recommendations, and displaying them. It integrates all other components.

## How to Use

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/movie-recommender.git
   cd movie-recommender
   ```

2. **Run the program**:
   ```bash
   python movie_recommender.py
   ```

3. **Rate Movies**: You will be prompted to rate movies from a predefined list. After you rate at least one movie, the system will provide you with recommendations based on similar users' ratings.

## Example Usage

```
Welcome to the Movie Recommender!
Please rate movies to get recommendations.

List of movies:
1. The Shawshank Redemption (1994)
2. The Godfather (1972)
...

Enter the movie number to rate (or 'done' to finish): 1
Enter your rating for 'The Shawshank Redemption' (1-5): 5
You rated 'The Shawshank Redemption' with a 5/5.

Enter the movie number to rate (or 'done' to finish): done

Recommendations for you:
The Godfather (Predicted Rating: 4.5/5)
Fight Club (Predicted Rating: 4.3/5)
```

## Data

The movie data used in this project is a hardcoded list of popular movies with their release year and titles. User data is simulated by predefined user ratings, which are updated as the current user adds their ratings.

## Contributions

If you’d like to contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
