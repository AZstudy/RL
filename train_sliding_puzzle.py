import gym
import os 
import dill

from baselines import deepq


def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = len(lcl['episode_rewards']) > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100.0 >= -20
    #return is_solved
    return False


def main():
    env = gym.make("SlidingPuzzle-v0")
    env.shuffle = 4
    model = deepq.models.mlp([20, 10, 5])

    # pretrained model
    model_file = "sliding_puzzle.pkl"
    model_path = os.path.realpath(model_file)

    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=1,
        pretrained_model_path=model_path,
        initial_p=1,
        callback=callback
    )
    
    print("Saving model to ", model_file)
    act.save(model_path)

if __name__ == '__main__':
    main()
