import pandas as pd
from star_generator import StarGenerator, Evaluation
import os
from star_generator_cleaning import CleanText
os.makedirs('batches', exist_ok=True)

# Example data load function

def update_model(batch_file, week):

    star_gen = StarGenerator()
    df = pd.read_csv(batch_file)
    X = df['review/text']  
    y = df['review/score'] 

    # Continuously update the model with new batches
    # for week in range(1, 10):  # simulate 10 weeks of updates
    cleaner = CleanText()
    X = X.apply(cleaner)
    corpus=X.tolist()
    star_gen.partial_train(corpus, y)
    
    if week % 5 == 0:  # Save the model every 5 weeks
        star_gen.save(f'saved_model/star_generator/star_generator_w{week}.pkl', f'saved_model/star_generator/vectorizer_star_generator_w{week}.pkl')
    
        #write a script to decide between the two models saving the one who has the best accuracy
    print('model is partially trained successfully')

    # # Optional: Evaluate the model periodically => this will be done in a dashboard
    # if week % 10 == 0:
    #     X_test, y_test = load_new_data(f"test_batch_{week}.csv")
    #     predictions = star_gen.predict(X_test)
    #     eval = Evaluation(y_test, predictions)
    #     eval.calculate_metrics()
    #     eval.print_metrics()
