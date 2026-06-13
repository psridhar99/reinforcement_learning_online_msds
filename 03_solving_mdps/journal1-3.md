```markdown
Journal Entry: Modules 1-3

The first thing that clicked for me is how RL is fundamentally different from supervised and unsupervised learning, in that data isn't inherently strucured or fed to the neural net in a tabular format like prior ML methods. The core training loop of RL is a tuple ofstate, action, reward, and next state. With RL, the agent has to discover the optimal policy through mapping states to actions and learning from rewards/penalties.

The k-armed bandit problem was unique to me given how it was a use case of RL where the state space was discrete and the action space was a single dimension. 

The ε-greedy approach balances this was interesting in that it exemplified the exploration and exploitation tradeoff. This also connected with hyperparameter tuning and entropy coefficient and how we balance action exploration and exploitation. 

MDPs formalized and connected to the output from gymnasium environments with its 4-tuple (SAPR) and connecting the idea that the current state is the only thing that matters for the future state and not the history. 

When I started this class, I had done RL before understanding the overall process, how to define and scale actions, observation spaces, and rewards, but now I have a much better understanding of the overall process since I can fundamentally appreciate the math behind exploitation vs exploration, the k-armed bandit problem, and the theory behind MDPs. It definitely gives me more insight into future RL problems, how to approach them, and even the difference w/on and off policies.


```