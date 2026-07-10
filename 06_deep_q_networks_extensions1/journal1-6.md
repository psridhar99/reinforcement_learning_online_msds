```markdown
Student: Pranav Sridhar (ps2pw)

Journal Entry: Modules 1-6
---------------------------------
The first thing that clicked for me is how RL is fundamentally different from supervised and unsupervised learning, in that data isn't inherently strucured or fed to the neural net in a tabular format like prior ML methods. The core training loop of RL is a tuple ofstate, action, reward, and next state. With RL, the agent has to discover the optimal policy through mapping states to actions and learning from rewards/penalties.

The k-armed bandit problem was unique to me given how it was a use case of RL where the state space was discrete and the action space was a single dimension. 

The ε-greedy approach balances this was interesting in that it exemplified the exploration and exploitation tradeoff. This also connected with hyperparameter tuning and entropy coefficient and how we balance action exploration and exploitation. 

MDPs formalized and connected to the output from gymnasium environments with its 4-tuple (SAPR) and connecting the idea that the current state is the only thing that matters for the future state and not the history. 

When I started this class, I had done RL before understanding the overall process, how to define and scale actions, observation spaces, and rewards, but now I have a much better understanding of the overall process since I can fundamentally appreciate the math behind exploitation vs exploration, the k-armed bandit problem, and the theory behind MDPs. It definitely gives me more insight into future RL problems, how to approach them, and even the difference w/on and off policies.

Then we got into temporal-difference learning, which was a middle ground between Monte Carlo and Dynamic Programming - it learns from experience like MC does, but it also bootstraps off its own estimates like DP does, so you don't have to wait until the end of an episode to update anything. Q-Learning was the algorithm we actually used it for, and the thing that stuck with me was that it's off-policy, meaning the agent can be acting all exploratory and random but still be learning the actual optimal policy underneath. The cliff walk lab made this click for me since I could actually watch the Q-table update.

From there tabular Q-Learning kind of falls apart pretty quickly when you have a huge state or obs space. Once your state space gets big or continuous, you just can't keep a table for every state-action pair anymore which is where deep learning comes in to help approximate Q(state, action) instead, but introduces some instability.The two things that stood out were experience replay (just keep a buffer of past transitions and sample randomly from it instead of learning from consecutive correlated steps) and target networks (freeze a copy of the network for a bit so you're not trying to hit a target that's shifting under you every update).

After that it was basically more patching on top of DQN. Prioritized experience replay is the same replay buffer idea, except instead of sampling uniformly you weight it so transitions with bigger TD error get replayed more since those are the ones you learn the most from. Double Q-Learning was a newer topic I learned which fixes the overestimation problem you get from using the same network to both pick the best action and score it, by splitting that into two networks instead. 

Looking back at it all, it's basically a story of going from an exact but limited method (Q-Learning) to a scalable but messier one (DQN), and then a bunch of follow-up papers cleaning up the mess that came with it. Makes it a lot easier to read deep RL papers now since I can tell most of them are targeted fixes rather than a whole new idea.
```
