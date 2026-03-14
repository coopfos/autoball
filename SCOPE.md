Autoball is an adaptation of Andrej Karpathy's autoresearch architecture for post-training small LLMs. I want to take the auto research workflow and apply it to trying to develop a sophisticated MLB game prediction model. Here is the autoresearch github: https://github.com/karpathy/autoresearch

The first step of this project is to scrape all of the MLB data from the 2025 season, and possibly earlier seasons. I need to build a scrape workflow from baseball reference to get this data.

Once I have a clean training data set, I will build some basic preliminary models to set the first benchmark. From that point forward, I will turn on the autoresearcher to work with the training set and tune models to try to narrow in on the best prediction model possible.

The 'autoball' architecture will differ from Karpathy's in some regards: Karpathy has a single defined evaluation metric, I have several options to pick from and am not necessarily confined to one single. Autoresearch has a hard 5-min cutoff per training run, most of the autoball model runs will not consume 5 minutes and there is no reason to have a hard set time for training besides a heartbeat to make sure the agent does not get stuck on a single research iteration.

The autoball agent will need to be able to make wideranging decisions in research, including: eval metric selection, model architecture selection, ensemble model weighting, data set subsetting, feature selection, feature engineering, etc... Really the autoball researcher needs to have full ownership of the feeder data, model building, and evaluation during research runs. This is human-out-of-the-loop researching.
