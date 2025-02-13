import argparse
import numpy as np


def parse_arguments():
    # parse arguments
    parser = argparse.ArgumentParser(description='Read .dat file for photogrammetry data.')
    parser.add_argument('file_path', nargs='?', type=str, default='..\\data\\lor49.dat',
                        help='The path to the .dat file')
    return parser.parse_args()


def read_dat_file(file_path):
    # read lor.dat and storage as dictionary
    data = {}
    points_list = []
    with open(file_path, 'r') as file:
        header = file.readline().split()
        data['point_count'] = int(header[1])
        data['angle_element_tolerance'] = float(header[2])
        data['line_element_tolerance'] = float(header[3])
        data['focal_length'] = float(header[4])
        data['principal_point_x'] = float(header[5])
        data['principal_point_y'] = float(header[6])
        data['unknown'] = int(header[7])
        
        for _ in range(data['point_count']):
            point_info = file.readline().split()
            point_data = [
                int(point_info[0]),
                float(point_info[1]),
                float(point_info[2]),
                float(point_info[3]),
                float(point_info[4]),
                float(point_info[5]),
            ]
            points_list.append(point_data)
    
    data['points'] = np.array(points_list)
    
    return data


def convert_to_image_plane(principal_point_x, principal_point_y, points):
    # convert control points from image coordinate to image plane coordinate
    pixel_coords = points[:, -2:]
    image_plane_coords = np.zeros_like(pixel_coords)
    image_plane_coords[:, 0] = pixel_coords[:, 0] - principal_point_x
    image_plane_coords[:, 1] = principal_point_y - pixel_coords[:, 1]
    return image_plane_coords


def calculate_scale(control_space_coords, control_img_coords, n):
    # calculate the scale of image
    if n < 2:
        raise ValueError("需要至少两个点来计算比例尺")

    sum_of_ratios = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            D = np.linalg.norm(control_space_coords[i] - control_space_coords[j])
            d = np.linalg.norm(control_img_coords[i] - control_img_coords[j])
            sum_of_ratios += D / d

    num_combinations = n * (n - 1) / 2

    return sum_of_ratios / num_combinations


def pixel_to_meter(dpi):
    # calculate the ratio of pixel to meter given dpi
    inches_to_meters = 0.0254
    return inches_to_meters / dpi


def update_units(data, pixel_to_meter_ratio):
    # transform data unit from pixel to meter
    data['focal_length'] *= pixel_to_meter_ratio
    data['principal_point_x'] *= pixel_to_meter_ratio
    data['principal_point_y'] *= pixel_to_meter_ratio

    data['points'][:, 4:] *= pixel_to_meter_ratio


def calculate_initial_exterior_line_elements(control_space_coords, scale, focal_length):
    # calculate initial exterior line elements
    Xs0 = np.mean(control_space_coords[:, 0])
    Ys0 = np.mean(control_space_coords[:, 1])
    Zs0 = np.mean(control_space_coords[:, 2]) + scale * focal_length

    return Xs0, Ys0, Zs0


def calculate_initial_distances_to_control_points(control_img_coords, scale, focal_length):
    # calculate the initial distance of S to control points
    distances = np.sqrt(control_img_coords[:, 0] ** 2 +
                        control_img_coords[:, 1] ** 2 +
                        focal_length ** 2)

    scaled_distances = scale * distances
    return scaled_distances


def calculate_distances_matrix(control_space_coords, n):
    # calculate the distances between control points
    distances_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            distance = np.linalg.norm(control_space_coords[i] - control_space_coords[j])
            distances_matrix[i, j] = distance
            distances_matrix[j, i] = distance

    return distances_matrix


def calculate_angles_and_factors(control_img_coords, focal_length, n):
    # calculate theta_ij, F_ij and G_ij
    cos_theta_matrix = np.zeros((n, n))
    F_matrix = np.zeros((n, n))
    G_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            xi, yi = control_img_coords[i]
            xj, yj = control_img_coords[j]
            denominator_i = np.sqrt(xi ** 2 + yi ** 2 + focal_length ** 2)
            denominator_j = np.sqrt(xj ** 2 + yj ** 2 + focal_length ** 2)
            numerator = xi * xj + yi * yj + focal_length ** 2
            cos_theta = numerator / (denominator_i * denominator_j)
            sin_theta_half_squared = max((1 - cos_theta) / 2, 1e-10)

            F_ij = 1 / (4 * sin_theta_half_squared)
            G_ij = F_ij * cos_theta

            cos_theta_matrix[i, j] = cos_theta
            cos_theta_matrix[j, i] = cos_theta
            F_matrix[i, j] = F_ij
            F_matrix[j, i] = F_ij
            G_matrix[i, j] = G_ij
            G_matrix[j, i] = G_ij

    return cos_theta_matrix, F_matrix, G_matrix


def iterate_to_solve_S(initial_S, distances_matrix, F_matrix, G_matrix, n):
    # calculate distance of S to control points
    num_equations = n * (n - 1) // 2
    A = np.zeros((num_equations, n))
    a = np.zeros((num_equations, 1))
    b = np.zeros((num_equations, 1))

    equation_idx = 0
    # control F to drop i and j
    min_F = np.min(F_matrix + np.eye(n) * np.max(F_matrix))
    
    for i in range(n):
        for j in range(i + 1, n):
            # check the condition
            if F_matrix[i, j] / min_F < 100:
                A[equation_idx, i] = 1
                A[equation_idx, j] = 1
                a[equation_idx] = 2 * F_matrix[i, j] * (distances_matrix[i, j]) ** 2
                b[equation_idx] = -2 * G_matrix[i, j] * (initial_S[i] - initial_S[j]) ** 2
                equation_idx += 1

    A = A[:equation_idx, :]
    a = a[:equation_idx]
    b = b[:equation_idx]

    S = np.copy(initial_S)
    iter_count = 0
    while True:
        iter_count += 1
        S_prev = np.copy(S)
        S2 = np.linalg.lstsq(A, a + b, rcond=None)[0].flatten()
        S = np.sqrt(S2)
        b_update_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if F_matrix[i, j] / min_F < 100:
                    b[b_update_idx] = -2 * G_matrix[i, j] * (S[i] - S[j]) ** 2
                    b_update_idx += 1

        if np.max(np.abs(S - S_prev)) < 1e-3:
            break

    print("解算S的迭代次数：", iter_count)

    return S


def calculate_observation_angles(control_img_coords, focal_length):
    # calculate observation angles
    alphas = []
    betas = []

    for xi, yi in control_img_coords:
        tan_alpha = xi / focal_length
        tan_beta = yi / np.sqrt(focal_length**2 + xi**2)

        alpha = np.arctan(tan_alpha)
        beta = np.arctan(tan_beta)

        alphas.append(alpha)
        betas.append(beta)

    return alphas, betas


def UVW_to_XYZ(UVWabc):
    # convert U V W to X Y Z
    U, V, W, a, b, c = UVWabc
    denom = -1 / (1 + a**2 + b**2 + c**2)
    inv_matrix = np.array([[1 + a**2, a*b + c, a*c - b],
                            [a*b - c, 1 + b**2, b*c + a],
                            [a*c + b, b*c - a, 1 + c**2]])
    XYZ = denom * inv_matrix @ np.array([U, V, W])
    return XYZ


def solve_exterior_elements(control_space_coords, S, alphas, betas, n):
    # solve U V W a b c
    B = np.zeros((3*n, 6))
    L = np.zeros((3*n, 1))

    for i in range(n):
        Xi, Yi, Zi = control_space_coords[i]
        Si = S[i]
        alphai = alphas[i]
        betai = betas[i]

        B[3*i:3*i+3, :] = [[1, 0, 0, 0, Si * np.cos(betai) * np.cos(alphai) + Zi, -Si * np.sin(betai) - Yi],
                           [0, 1, 0, -Si * np.cos(betai) * np.cos(alphai) - Zi, 0, Si * np.cos(betai) * np.sin(alphai) + Xi],
                           [0, 0, 1, Si * np.sin(betai) + Yi, -Si * np.cos(betai) * np.sin(alphai) - Xi, 0]]
        L[3*i:3*i+3, :] = [[Si * np.cos(betai) * np.sin(alphai) - Xi],
                           [Si * np.sin(betai) - Yi],
                           [Si * np.cos(betai) * np.cos(alphai) - Zi]]

    X, _, _, _ = np.linalg.lstsq(B, L, rcond=None)
    XYZ = UVW_to_XYZ(X.flatten())
    XYZ[2] = abs(XYZ[2])
    abc = X.flatten()[3:]
    return XYZ, abc


def iterate_to_solve_exterior_elements(control_space_coords, S, alphas, betas, n):
    # solve U V W a b c iteratively
    prev_XYZ = np.zeros(3)
    prev_abc = np.zeros(3)
    iteration = 0

    while True:
        iteration += 1
        # solve U V W a b c
        B = np.zeros((3*n, 6))
        L = np.zeros((3*n, 1))

        for i in range(n):
            Xi, Yi, Zi = control_space_coords[i]
            Si = S[i]
            alphai = alphas[i]
            betai = betas[i]

            B[3*i:3*i+3, :] = [[1, 0, 0, 0, Si * np.cos(betai) * np.cos(alphai) + Zi, -Si * np.sin(betai) - Yi],
                               [0, 1, 0, -Si * np.cos(betai) * np.cos(alphai) - Zi, 0, Si * np.cos(betai) * np.sin(alphai) + Xi],
                               [0, 0, 1, Si * np.sin(betai) + Yi, -Si * np.cos(betai) * np.sin(alphai) - Xi, 0]]
            L[3*i:3*i+3, :] = [[Si * np.cos(betai) * np.sin(alphai) - Xi],
                               [Si * np.sin(betai) - Yi],
                               [Si * np.cos(betai) * np.cos(alphai) - Zi]]

        UVWabc, _, _, _ = np.linalg.lstsq(B, L, rcond=None)
        UVWabc = UVWabc.flatten()
        XYZ = UVW_to_XYZ(UVWabc)

        for i in range(n):
            Xi, Yi, Zi = control_space_coords[i]
            S[i] = np.sqrt((XYZ[0] - Xi)**2 + (XYZ[1] - Yi)**2 + (XYZ[2] - Zi)**2)

        if iteration > 1 and np.all(np.abs(XYZ - prev_XYZ) < 0.1) and np.all(np.abs(UVWabc[3:] - prev_abc) < 5e-5):
            break

        prev_XYZ = XYZ
        prev_abc = UVWabc[3:]
    
    XYZ[2] = abs(XYZ[2])
    print("求解外方位元素迭代次数：", iteration)

    return XYZ, UVWabc[3:]


def rotation_matrix_from_angles(a, b, c):
    # calculate rotation matrix from angles
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(a), -np.sin(a)],
                    [0, np.sin(a), np.cos(a)]])
    
    R_y = np.array([[np.cos(b), 0, np.sin(b)],
                    [0, 1, 0],
                    [-np.sin(b), 0, np.cos(b)]])
    
    R_z = np.array([[np.cos(c), -np.sin(c), 0],
                    [np.sin(c), np.cos(c), 0],
                    [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def compute_pixel_residuals(control_space_coords, control_img_coords, XYZ, R, f):
    # calculate pixel residuals
    Xs, Ys, Zs = XYZ
    pixel_residuals = {'Delta_x': [], 'Delta_y': []}

    for i, (Xi, Yi, Zi) in enumerate(control_space_coords):
        img_x, img_y = control_img_coords[i]

        denom = R[2, 0] * (Xi - Xs) + R[2, 1] * (Yi - Ys) + R[2, 2] * (Zi - Zs)
        Delta_x = -f * (R[0, 0] * (Xi - Xs) + R[0, 1] * (Yi - Ys) + R[0, 2] * (Zi - Zs)) / denom - img_x
        Delta_y = -f * (R[1, 0] * (Xi - Xs) + R[1, 1] * (Yi - Ys) + R[1, 2] * (Zi - Zs)) / denom - img_y

        pixel_residuals['Delta_x'].append(Delta_x)
        pixel_residuals['Delta_y'].append(Delta_y)

    return pixel_residuals


def main():
    # read file
    args = parse_arguments()
    data = read_dat_file(args.file_path)

    # unify unit
    dpi = 51
    pixel_to_meter_ratio = pixel_to_meter(dpi)
    update_units(data, pixel_to_meter_ratio)

    # unify coordinates
    control_space_coords = data['points'][:, 1:4]
    control_img_coords = convert_to_image_plane(data['principal_point_x'], data['principal_point_y'], data['points'])

    # calculate initial scale and exterior line elements
    scale = calculate_scale(control_space_coords, control_img_coords, data['point_count'])
    Xs0, Ys0, Zs0 = calculate_initial_exterior_line_elements(control_space_coords, scale, data['focal_length'])

    # calculate distance from S to control points
    initial_S = calculate_initial_distances_to_control_points(control_img_coords, scale, data['focal_length'])
    distances_matrix = calculate_distances_matrix(control_space_coords, data['point_count'])
    cos_theta_matrix, F_matrix, G_matrix = calculate_angles_and_factors(control_img_coords, data['focal_length'], data['point_count'])
    S = iterate_to_solve_S(initial_S, distances_matrix, F_matrix, G_matrix, data['point_count'])

    # calculate exterior elements
    alphas, betas = calculate_observation_angles(control_img_coords, data['focal_length'])
    XYZ, abc = solve_exterior_elements(control_space_coords, S, alphas, betas, data['point_count'])
    XYZ_iter, abc_iter = iterate_to_solve_exterior_elements(control_space_coords, S, alphas, betas, data['point_count'])
    
    # calculate residuals
    R = rotation_matrix_from_angles(*abc)
    pixel_residuals = compute_pixel_residuals(control_space_coords, control_img_coords, XYZ, R, data['focal_length'])
    R_iter = rotation_matrix_from_angles(*abc_iter)
    pixel_residuals_iter = compute_pixel_residuals(control_space_coords, control_img_coords, XYZ_iter, R_iter, data['focal_length'])

    # adjusted_pixel_residuals = {key: [value / pixel_to_meter_ratio for value in values] for key, values in pixel_residuals.items()}
    # adjusted_pixel_residuals_iter = {key: [value / pixel_to_meter_ratio for value in values] for key, values in pixel_residuals.items()}

    print("外方位线元素单次解：", XYZ)
    print("外方位角元素单次解：", abc)
    print("单次解像素残差：", pixel_residuals)

    print("外方位线元素迭代解：", XYZ_iter)
    print("外方位角元素迭代解：", abc_iter)
    print("迭代解像素残差：", pixel_residuals_iter)


if __name__ == '__main__':
    main()
