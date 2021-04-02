using .A2funcs: log1pexp # log(1 + exp(x)) stable
using .A2funcs: factorized_gaussian_log_density
using .A2funcs: skillcontour!
using .A2funcs: plot_line_equal_skill!


#Q4
#alternative way of finding games
function find_indrank(name)
  ind = [i for i in 1: num_players if player_names[i] == name][1]
  return (ind,perm[ind])
end
find_indrank("Roger-Federer")
find_indrank("Novak-Djokovic")
RF_ind = [i for i in 1: num_players if player_names[i] == "Roger-Federer"][1]
RN_ind = [i for i in 1: num_players if player_names[i] == "Rafael-Nadal"][1]
RFRN_game_ind = [k for k in 1:length(tennis_games[:,1]) if
              (RF_ind in tennis_games[k,:] && RN_ind in tennis_games[k,:])]
RFRN_games = tennis_games[RFRN_game_ind,:]

function find_games(p1, p2)
  ind1 = find_player_info(p1)[1]
  ind2 = find_player_info(p2)[1]
  game_ind = [i for i in 1:num_games if ind1
              in tennis_games[i,:] && ind2 in tennis_games[i,:]]
  return tennis_games[game_ind,:]
end
find_games("Rafael-Nadal", "Roger-Federer")
count_wins = sum([1 for i in 1:size(RFRN_games)[1] if RFRN_games[i,1] == 1])
game = two_player_toy_games(count_wins, size(RFRN_games)[1] - count_wins)
