clear;
clc;

reward = [-0.048 0.036 0.054 0.236 0.028 0.014 0.014 0.024 0.134 0.056 0.124 0.268 0.1 0.152 0.194 0.432 0.368 0.522 0.602 0.646 0.682 0.726 0.694 0.154 0.932 0.922 0.876 0.92 0.882 0.948 0.352 0.962 0.896 0.844 0.912 0.91 0.934 0.884 0.86 0.898 0.862 0.844 0.858 0.842 0.884 0.844 0.822 0.822 0.824 0.846 0.786 0.796 0.834 0.752 0.77 0.788 0.738 0.844 0.634 0.804 0.708 0.782 0.736 0.77 0.688 0.742 0.75 0.734];
length(reward);
episodes = 250:250:17000;
length(episodes);

figure;
plot(episodes,reward)
xlabel('Episodes');
ylabel('Mean episode reward');
title("Mean episode reward every 250 episodes");

%%
 
figure;
square = [55 69 66 75];
names_square = categorical({'5x5','7x7','10x10','15x15'});

% Specify the desired order of categories
new_order = {'5x5', '7x7', '10x10', '15x15'};

% Reorder the categories
names_square = reordercats(names_square, new_order);

bar(names_square,square)
xlabel('Board size');
ylabel('Number of wins');
title("Number of wins per board size (square) for 100 games");


%% 

figure;
square = [56 59 63 72];
names_square = categorical({'2x5','7x5','9x12','5x15'});

% Specify the desired order of categories
new_order = {'2x5', '7x5', '9x12', '5x15'};

% Reorder the categories
names_square = reordercats(names_square, new_order);

bar(names_square,square)
xlabel('Board size');
ylabel('Number of wins');
title("Number of wins per board size (non-square) for 100 games");
