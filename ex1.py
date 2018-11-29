import gym
import random 
import numpy as np
from statistics import mean, median
from collections import Counter

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam




LR = 1e-3
env = gym.make('CartPole-v0')
env.reset()
goal_steps = 500
score_requirement = 50
initial_games = 20000 #10000


def initial_population():
        training_data = []
        scores = []
        accepted_scores = []
        for _ in range(initial_games):
                env.reset()
                score = 0
                game_memory = []
                prev_observation = []
                for _ in range(goal_steps):
                        action = random.randrange(0, 2)
                        observation, reward, done, info = env.step(action)

                        if done:
                                break

                        if len(prev_observation) > 0:
                                game_memory.append([prev_observation, action])

                        prev_observation = observation
                        score += reward

                if score >= score_requirement:
                        accepted_scores.append(score)

                        for data in game_memory:
                                if data[1] == 1:
                                        output = [0, 1]
                                elif data[1] == 0:
                                        output = [1,0]

                                training_data.append([data[0], output])

                scores.append(score)
        training_data_save = np.array(training_data)
        np.save('saved.npy', training_data_save)
        print('Average accepted_scores:', mean(accepted_scores))
        print('Median accepted_scores:', median(accepted_scores))

        env.close()

        return training_data

# print(initial_population())


def network_model(input_shape):

        model = Sequential()

        model.add(Dense(2, activation='relu', input_dim=input_shape))
        model.add(Dropout(0.8))

        model.add(Dense(2, activation='relu'))
        model.add(Dropout(0.8))

        model.add(Dense(2, activation='relu'))
        model.add(Dropout(0.8))

        model.add(Dense(2, activation='relu'))
        model.add(Dropout(0.8))

        model.add(Dense(2, activation='relu'))
        model.add(Dropout(0.8))

        model.add(Dense(2, activation='softmax'))

        adam = Adam(lr=0.01, decay=1e-6)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        return model

def train_model(training_data, model=False):
        X = np.stack(np.array([i[0] for i in training_data])) #.reshape(-1, 4)
        y = np.array([i[1] for i in training_data])
        print(len(X[0]))
        if not model:
                model = network_model(input_shape=len(X[0]))
        
        tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=False)
        model.fit(X, y,
          epochs=2,
          batch_size=32,
          callbacks=[tbCallBack])
        return model

training_data = np.load('saved.npy')
model = train_model(training_data)



def test_model(model):
        choices = []
        scores = []
        for episode in range(initial_games):
                score = 0
                game_memory = []
                prev_obs = []
                env.reset()

                for _ in range(goal_steps):
                        env.render()
                        if len(prev_obs) > 0:
                                pred_action = np.argmax(model.predict(prev_obs.reshape(1,4)))
                        else:
                                pred_action = random.randrange(0,2)

                        observation, reward, done, info = env.step(pred_action)
                        prev_obs = observation
                        choices.append(pred_action)
                        game_memory.append([prev_obs, pred_action])
                        score += reward
                        if done:
                                break
                scores.append(score)
                print('Average score :', mean(scores))
                print('Median score :', median(scores))

        env.close()

test_model(model)




