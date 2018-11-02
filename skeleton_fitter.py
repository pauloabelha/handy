from skeleton import *

#animate_skeleton()

Theta_lims = get_Theta_lims()
bones_lengths = get_bones_lengths()
fingers_angles = get_fingers_angles_canonical()

Theta = np.array([0.1] * 23)
#print(Theta)
hand_seq = get_hand_seq(Theta, bones_lengths, fingers_angles)
#plot_bone_lines(hand_seq)

hand_matrix = Theta_to_hand_matrix(Theta, bones_lengths, fingers_angles)
#print(hand_matrix)

target_matrix = get_example_target_matrix2()
#print(target_matrix)
#plot_hand_matrix(target_matrix)

loss = E_pos3D(Theta, target_matrix, bones_lengths, fingers_angles)
print(loss)

Theta_fit, losses = fit_skeleton(Epsilon_Loss, target_matrix, bones_lengths, fingers_angles, Theta_lims,
                         initial_theta=Theta, num_iter=500, log_interval=50, lr=7.5e-5)
hand_seq_fit = get_hand_seq(Theta_fit, bones_lengths, fingers_angles)

hand_matrix = Theta_to_hand_matrix(Theta_fit, bones_lengths, fingers_angles)
print(Theta_fit)

print('Average 3D joint error (mm): {}'.format(get_avg_3D_error(hand_matrix, target_matrix)))

#fig = plot_hand_matrix(target_matrix, show=False)
#plot_hand_matrix(hand_matrix, fig=fig)
#plot_bone_lines(hand_seq_fit)




