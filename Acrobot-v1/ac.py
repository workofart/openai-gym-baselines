import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import gym
import time

from actor import Actor
from critic import Critic


def plot(i, total_rewards, final_plot_i = 1000, plot_frequency = 100):
    if i > 0 and i % plot_frequency == 0:
        print(f'EP[{i}]: {total_rewards[-1]}') # raw total rewards for that episode
    
    if i == final_plot_i:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_ylabel('Episode Reward')
        ax.set_xlabel('Episode')
        ax.plot(total_rewards)
        plt.show()

def test(env, actor):
    print('TEST RUN!')
    for i in range(10):
        state = env.reset()
        done = False
        num_steps = 0
        while not done:
            action = actor.act(state.reshape((1, 6)))
            state, reward, done, _  = env.step(action)
            env.render()
            num_steps += 1
        print(num_steps)
        input('wait')
        state = env.reset()
        env.close()

# hyperparameters
LR_A = 1e-4   # learning rate for actor
LR_C = 5e-3     # learning rate for critic (should learn faster)
num_episodes = 250
plot_frequency = 10

# Network params
N_STATES = 6
N_ACTION = 3

# Reproducibility
np.random.seed(2)
tf.random.set_random_seed(2)

def test_dummy(env):
    """ Utility function for testing the environment using a random agent,
        no training involved, pure random actions

    Args:
        env (gym.env): OpenAI Gym Environment
    """
    state = env.reset()
    plotting_rewards = []
    actions = [0, 2]
    for i in range(num_episodes+1):
        done = False
        rewards = 0
        states = []
        state = env.reset()
        step = 0
        while not done:
            state = state.reshape((1, 6))
            action = np.random.choice(actions)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape((1,6))
            state = next_state

            # Book keeping
            rewards += reward
            step += 1
        plotting_rewards.append(rewards)
        
        # Optional reward smoothing using a moving average window
        # moving_rewards = np.convolve(plotting_rewards, np.ones((20,)) / 20, mode='valid')
        
        plot(i, moving_rewards, num_episodes - 1, plot_frequency=plot_frequency)  

def train(env, sess, actor, critic):
    """ Training function that allows the agent to interact with the environment


    Args:
        env (gym.env): OpenAI Gym environment
        sess (tf.session): Tensorflow Session
        actor (Actor): The actor class for actor-critic agent representation
        critic (Critic): The critic class for actor-critic agent representation
    """
    start_time = time.time()
    state = env.reset()
    plotting_rewards = []
    for i in range(num_episodes+1):
        done = False
        rewards = 0
        state = env.reset()
        while not done:
            state = state.reshape((1, 6))
            action = actor.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape((1,6))

            advantage = critic.update(state, reward, next_state)
            actor.update(state, action, advantage)
            state = next_state

            # Book keeping
            rewards += reward
        plotting_rewards.append(rewards)

        # Optional reward smoothing using a moving average window
        # moving_rewards = np.convolve(plotting_rewards, np.ones((20,)) / 20, mode='valid')

        plot(i, plotting_rewards, num_episodes - 1, plot_frequency=plot_frequency)        
    print(f'Training took: {(time.time() - start_time):.2f} seconds')

if __name__ == '__main__':
    sess = tf.Session()
    

    actor = Actor(sess, n_features=N_STATES, n_actions=N_ACTION, lr=LR_A)
    critic = Critic(sess, n_features=N_STATES, lr=LR_C)

    sess.run(tf.global_variables_initializer())

    env = gym.make('Acrobot-v1')
    env.seed(2) # reproducibility
    env.reset()

    train(env, sess, actor, critic)
    # test(env, actor)

