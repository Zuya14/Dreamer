import time
import os

import gym
import pybullet_envs  # PyBulletの環境をgymに登録する

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter


from agent import Agent
from model import Encoder, RecurrentStateSpaceModel, ObservationModel, RewardModel, ValueModel, ActionModel
from utils import ReplayBuffer, preprocess_obs, lambda_target
from wrappers import make_env

import argparse


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--buffer-capacity', type=int, default=200000) # 元実装 1000000

    parser.add_argument('--state-dim', type=int, default=30) # 確率的状態の次元
    parser.add_argument('--rnn-hidden-dim', type=int, default=200) # 決定的状態（RNNの隠れ状態）の次元

    parser.add_argument('--model_lr', type=float, default=6e-4) # encoder, rssm, obs_model, reward_modelの学習率
    parser.add_argument('--value_lr', type=float, default=8e-5)
    parser.add_argument('--action_lr', type=float, default=8e-5)
    parser.add_argument('--eps', type=float, default=1e-4)


    parser.add_argument('--env-name', type=str)
    parser.add_argument('--log-dir', type=str, default='log')

    parser.add_argument('--seed-episodes', type=int, default=5)        # 最初にランダム行動で探索するエピソード数
    parser.add_argument('--all-episodes', type=int, default=100)       # 学習全体のエピソード数
    parser.add_argument('--test-interval', type=int, default=10)       # 何エピソードごとに探索ノイズなしのテストを行うか
    parser.add_argument('--model-save-interval', type=int, default=20) # NNの重みを何エピソードごとに保存するか
    parser.add_argument('--collect-interval', type=int, default=100)   # 何回のNNの更新ごとに経験を集めるか（＝1エピソード経験を集めるごとに何回更新するか）

    parser.add_argument('--action-noise-var', type=float, default=0.3) # 探索ノイズの強さ

    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--chunk-length', type=int, default=50)        # 1回の更新で用いる系列の長さ
    parser.add_argument('--imagination-horizon', type=int, default=15) # Actor-Criticの更新のために, Dreamerで何ステップ先までの想像上の軌道を生成するか

    parser.add_argument('--gamma', type=float, default=0.9)       # 割引率
    parser.add_argument('--lambda_', type=float, default=0.95)     # λ-returnのパラメータ
    parser.add_argument('--clip-grad-norm', type=int, default=100) # gradient clippingの値
    parser.add_argument('--free-nats', type=int, default=3)        # KL誤差（RSSMのpriorとposteriorの間の誤差）がこの値以下の場合, 無視する

    args = parser.parse_args()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)


    env = make_env(args.env_name)

    replay_buffer = ReplayBuffer(capacity=args.buffer_capacity,
                                observation_shape=env.observation_space.shape,
                                action_dim=env.action_space.shape[0])

    # モデルの宣言
    encoder = Encoder().to(device)
    rssm = RecurrentStateSpaceModel(args.state_dim,
                                    env.action_space.shape[0],
                                    args.rnn_hidden_dim).to(device)
    obs_model = ObservationModel(args.state_dim, args.rnn_hidden_dim).to(device)
    reward_model = RewardModel(args.state_dim, args.rnn_hidden_dim).to(device)

    value_model = ValueModel(args.state_dim, args.rnn_hidden_dim).to(device)
    action_model = ActionModel(args.state_dim, args.rnn_hidden_dim,
                                env.action_space.shape[0]).to(device)

    # オプティマイザの宣言
    model_params = (list(encoder.parameters()) +
                    list(rssm.parameters()) +
                    list(obs_model.parameters()) +
                    list(reward_model.parameters()))
    model_optimizer = torch.optim.Adam(model_params, lr=args.model_lr, eps=args.eps)
    value_optimizer = torch.optim.Adam(value_model.parameters(), lr=args.value_lr, eps=args.eps)
    action_optimizer = torch.optim.Adam(action_model.parameters(), lr=args.action_lr, eps=args.eps)


    writer = SummaryWriter(args.log_dir)


    for episode in range(args.seed_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, done, _ = env.step(action)
            replay_buffer.push(obs, action, reward, done)
            obs = next_obs

    for episode in range(args.seed_episodes, args.all_episodes):
        # -----------------------------
        #      経験を集める
        # -----------------------------
        start = time.time()
        # 行動を決定するためのエージェントを宣言
        policy = Agent(encoder, rssm, action_model)

        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            # 探索のためにガウス分布に従うノイズを加える
            action = policy(obs) + np.random.normal(0, np.sqrt(args.action_noise_var), env.action_space.shape[0])
            next_obs, reward, done, _ = env.step(action)
            replay_buffer.push(obs, action, reward, done)
            obs = next_obs
            total_reward += reward

        # 訓練時の報酬と経過時間をログとして表示
        writer.add_scalar('total reward at train', total_reward, episode)
        print('episode [%4d/%4d] is collected. Total reward is %f' %
                (episode+1, args.all_episodes, total_reward))
        print('elasped time for interaction: %.2fs' % (time.time() - start))

        # NNのパラメータを更新する
        start = time.time()
        for update_step in range(args.collect_interval):
            # -----------------------------------------------------------------
            #  encoder, rssm, obs_model, reward_modelの更新
            # -----------------------------------------------------------------
            observations, actions, rewards, _ = \
                replay_buffer.sample(args.batch_size, args.chunk_length)

            # 観測を前処理し, RNNを用いたPyTorchでの学習のためにTensorの次元を調整
            observations = preprocess_obs(observations)
            observations = torch.as_tensor(observations, device=device)
            observations = observations.transpose(3, 4).transpose(2, 3)
            observations = observations.transpose(0, 1)
            actions = torch.as_tensor(actions, device=device).transpose(0, 1)
            rewards = torch.as_tensor(rewards, device=device).transpose(0, 1)

            # 観測をエンコーダで低次元のベクトルに変換
            embedded_observations = encoder(
                observations.reshape(-1, 3, 64, 64)).view(args.chunk_length, args.batch_size, -1)

            # 低次元の状態表現を保持しておくためのTensorを定義
            states = torch.zeros(args.chunk_length, args.batch_size, args.state_dim, device=device)
            rnn_hiddens = torch.zeros(args.chunk_length, args.batch_size, args.rnn_hidden_dim, device=device)

            # 低次元の状態表現は最初はゼロ初期化
            state = torch.zeros(args.batch_size, args.state_dim, device=device)
            rnn_hidden = torch.zeros(args.batch_size, args.rnn_hidden_dim, device=device)

            # 状態予測を行ってそのロスを計算する（priorとposteriorの間のKLダイバージェンス）
            kl_loss = 0
            for l in range(args.chunk_length-1):
                next_state_prior, next_state_posterior, rnn_hidden = \
                    rssm(state, actions[l], rnn_hidden, embedded_observations[l+1])
                state = next_state_posterior.rsample()
                states[l+1] = state
                rnn_hiddens[l+1] = rnn_hidden
                kl = kl_divergence(next_state_prior, next_state_posterior).sum(dim=1)
                kl_loss += kl.clamp(min=args.free_nats).mean()  # 原論文通り, KL誤差がfree_nats以下の時は無視
            kl_loss /= (args.chunk_length - 1)

            # states[0] and rnn_hiddens[0]はゼロ初期化なので以降では使わない
            states = states[1:]
            rnn_hiddens = rnn_hiddens[1:]

            # 観測を再構成, また, 報酬を予測
            flatten_states = states.view(-1, args.state_dim)
            flatten_rnn_hiddens = rnn_hiddens.view(-1, args.rnn_hidden_dim)
            recon_observations = obs_model(flatten_states, flatten_rnn_hiddens).view(args.chunk_length-1, args.batch_size, 3, 64, 64)
            predicted_rewards = reward_model(flatten_states, flatten_rnn_hiddens).view(args.chunk_length-1, args.batch_size, 1)

            # 観測と報酬の予測誤差を計算
            obs_loss = 0.5 * F.mse_loss(recon_observations, observations[1:], reduction='none').mean([0, 1]).sum()
            reward_loss = 0.5 * F.mse_loss(predicted_rewards, rewards[:-1])

            # 以上のロスを合わせて勾配降下で更新する
            model_loss = kl_loss + obs_loss + reward_loss
            model_optimizer.zero_grad()
            model_loss.backward()
            clip_grad_norm_(model_params, args.clip_grad_norm)
            model_optimizer.step()

            # ----------------------------------------------
            #  Action Model, Value　Modelの更新
            # ----------------------------------------------
            # Actor-Criticのロスで他のモデルを更新することはないので勾配の流れを一度遮断
            flatten_states = flatten_states.detach()
            flatten_rnn_hiddens = flatten_rnn_hiddens.detach()

            # DreamerにおけるActor-Criticの更新のために, 現在のモデルを用いた
            # 数ステップ先の未来の状態予測を保持するためのTensorを用意
            imaginated_states = torch.zeros(args.imagination_horizon + 1,
                                            *flatten_states.shape,
                                            device=flatten_states.device)
            imaginated_rnn_hiddens = torch.zeros(args.imagination_horizon + 1,
                                                    *flatten_rnn_hiddens.shape,
                                                    device=flatten_rnn_hiddens.device)

            #　未来予測をして想像上の軌道を作る前に, 最初の状態としては先ほどモデルの更新で使っていた
            # リプレイバッファからサンプルされた観測データを取り込んだ上で推論した状態表現を使う
            imaginated_states[0] = flatten_states
            imaginated_rnn_hiddens[0] = flatten_rnn_hiddens
            
            # open-loopで未来の状態予測を使い, 想像上の軌道を作る
            for h in range(1, args.imagination_horizon + 1):
                # 行動はActionModelで決定. この行動はモデルのパラメータに対して微分可能で,
                #　これを介してActionModelは更新される
                actions = action_model(flatten_states, flatten_rnn_hiddens)
                flatten_states_prior, flatten_rnn_hiddens = rssm.prior(flatten_states,
                                                                    actions,
                                                                    flatten_rnn_hiddens)
                flatten_states = flatten_states_prior.rsample()
                imaginated_states[h] = flatten_states
                imaginated_rnn_hiddens[h] = flatten_rnn_hiddens

            # 予測された架空の軌道に対する報酬を計算
            flatten_imaginated_states = imaginated_states.view(-1, args.state_dim)
            flatten_imaginated_rnn_hiddens = imaginated_rnn_hiddens.view(-1, args.rnn_hidden_dim)
            imaginated_rewards = reward_model(flatten_imaginated_states, flatten_imaginated_rnn_hiddens).view(args.imagination_horizon + 1, -1)
            imaginated_values  = value_model(flatten_imaginated_states, flatten_imaginated_rnn_hiddens).view(args.imagination_horizon + 1, -1)
            # λ-returnのターゲットを計算
            lambda_target_values = lambda_target(imaginated_rewards, imaginated_values, args.gamma, args.lambda_)

            # TD(λ)ベースの目的関数で価値関数を更新
            value_loss = 0.5 * F.mse_loss(imaginated_values, lambda_target_values.detach().clone())
            value_optimizer.zero_grad()
            value_loss.backward(retain_graph=True)
            clip_grad_norm_(value_model.parameters(), args.clip_grad_norm)
            value_optimizer.step()

            # 価値関数の予測した価値が大きくなるようにActionModelを更新
            # PyTorchの基本は勾配降下だが, 今回は大きくしたいので-1をかける
            action_loss = -1 * (lambda_target_values.mean())
            action_optimizer.zero_grad()
            action_loss.backward()
            clip_grad_norm_(action_model.parameters(), args.clip_grad_norm)
            action_optimizer.step()

            # ログをTensorBoardに出力
            print('update_step: %3d model loss: %.5f, kl_loss: %.5f, '
                'obs_loss: %.5f, reward_loss: %.5f, '
                'value_loss: %.5f action_loss: %.5f'
                    % (update_step + 1, model_loss.item(), kl_loss.item(),
                        obs_loss.item(), reward_loss.item(),
                        value_loss.item(), action_loss.item()))
            total_update_step = episode * args.collect_interval + update_step
            writer.add_scalar('model loss', model_loss.item(), total_update_step)
            writer.add_scalar('kl loss', kl_loss.item(), total_update_step)
            writer.add_scalar('obs loss', obs_loss.item(), total_update_step)
            writer.add_scalar('reward loss', reward_loss.item(), total_update_step)
            writer.add_scalar('value loss', value_loss.item(), total_update_step)
            writer.add_scalar('action loss', action_loss.item(), total_update_step)

        print('elasped time for update: %.2fs' % (time.time() - start))

        # --------------------------------------------------------------
        #    テストフェーズ. 探索ノイズなしでの性能を評価する
        # --------------------------------------------------------------
        if (episode + 1) % args.test_interval == 0:
            policy = Agent(encoder, rssm, action_model)
            start = time.time()
            obs = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = policy(obs, training=False).reshape(env.action_space.shape)
                obs, reward, done, _ = env.step(action)
                total_reward += reward

            writer.add_scalar('total reward at test', total_reward, episode)
            print('Total test reward at episode [%4d/%4d] is %f' %
                    (episode+1, args.all_episodes, total_reward))
            print('elasped time for test: %.2fs' % (time.time() - start))

        if (episode + 1) % args.model_save_interval == 0:
            # 定期的に学習済みモデルのパラメータを保存する
            model_log_dir = os.path.join(args.log_dir, 'episode_%04d' % (episode + 1))
            os.makedirs(model_log_dir)
            torch.save(encoder.state_dict(), os.path.join(model_log_dir, 'encoder.pth'))
            torch.save(rssm.state_dict(), os.path.join(model_log_dir, 'rssm.pth'))
            torch.save(obs_model.state_dict(), os.path.join(model_log_dir, 'obs_model.pth'))
            torch.save(reward_model.state_dict(), os.path.join(model_log_dir, 'reward_model.pth'))
            torch.save(value_model.state_dict(), os.path.join(model_log_dir, 'value_model.pth'))
            torch.save(action_model.state_dict(), os.path.join(model_log_dir, 'action_model.pth'))

    writer.close()

if __name__ == '__main__':
    main()