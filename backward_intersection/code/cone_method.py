import numpy as np


def read_dat_file(file_path):
    with open(file_path, 'r') as file:
        org_data = list(map(float, file.readline().split()))
        base_data = np.array([list(map(float, line.split())) for line in file])
    return org_data, base_data


def solve_orientation(base_data, org_data):
    dpi = 0.498 / 1000
    m = 5000
    n_points = int(org_data[1])
    f = org_data[4] * dpi
    x0, y0 = org_data[5:7]
    X0, Y0 = np.mean(base_data[:, 1:3], axis=0)
    Z0 = m * f / 1000

    obj_X, obj_Y, obj_Z = base_data[:, 1:4].T
    img_x = base_data[:, 4] * dpi - x0 * dpi
    img_y = y0 * dpi - base_data[:, 5] * dpi

    # iterative solution for line elements
    Xs, Ys, Zs = X0, Y0, Z0
    iter_line = 0
    while True:
        iter_line += 1
        img_s = np.sqrt((img_x - x0)**2 + (img_y - y0)**2 + f**2)
        obj_s = np.sqrt((obj_X - Xs)**2 + (obj_Y - Ys)**2 + (obj_Z - Zs)**2)

        A, L = [], []
        for i in range(n_points):
            for j in range(i + 1, n_points):
                img_cos = ((x0 - img_x[i])*(x0 - img_x[j]) + (y0 - img_y[i])*(y0 - img_y[j]) + f**2) / (img_s[i] * img_s[j])
                obj_cos = ((Xs - obj_X[i])*(Xs - obj_X[j]) + (Ys - obj_Y[i])*(Ys - obj_Y[j]) + (Zs - obj_Z[i])*(Zs - obj_Z[j])) / (obj_s[i] * obj_s[j])
                T0 = (1 - obj_s[i] * obj_cos / obj_s[j]) / (obj_s[i] * obj_s[j])
                T1 = (1 - obj_s[j] * obj_cos / obj_s[i]) / (obj_s[i] * obj_s[j])
                a = (Xs - obj_X[i])*T1 + (Xs - obj_X[j])*T0
                b = (Ys - obj_Y[i])*T1 + (Ys - obj_Y[j])*T0
                c = (Zs - obj_Z[i])*T1 + (Zs - obj_Z[j])*T0
                l = img_cos - obj_cos
                A.append([a, b, c])
                L.append(l)

        A = np.array(A)
        L = np.array(L)
        Xnew = np.linalg.lstsq(A, L, rcond=None)[0].flatten()

        Xs += Xnew[0]
        Ys += Xnew[1]
        Zs += Xnew[2]

        if np.linalg.norm(Xnew) < float(org_data[4]):
            break
    
    print("求解线元素迭代次数：", iter_line)

    # direct solution for angle elements
    SP = np.sqrt((obj_X - Xs)**2 + (obj_Y - Ys)**2 + (obj_Z - Zs)**2)
    Sp = np.sqrt((img_x - x0)**2 + (img_y - y0)**2 + f**2)
    aa = (obj_X - Xs) / SP
    bb = (obj_Y - Ys) / SP
    cc = (obj_Z - Zs) / SP
    l1 = (img_x - x0) / Sp
    l2 = (img_y - y0) / Sp
    l3 = -f / Sp

    A = np.vstack([aa, bb, cc]).T
    R1 = np.linalg.lstsq(A, l1, rcond=None)[0].flatten()
    R2 = np.linalg.lstsq(A, l2, rcond=None)[0].flatten()
    R3 = np.linalg.lstsq(A, l3, rcond=None)[0].flatten()
    R = np.vstack([R1, R2, R3]).T

    phi = np.arctan2(-R3[0], R3[2])
    omega = np.arcsin(-R3[1])
    kappa = np.arctan2(R1[1], R2[1])

    return Xs, Ys, abs(Zs), phi, omega, kappa, R, np.array([img_x, img_y]).T, f


def compute_pixel_residuals(control_space_coords, control_img_coords, XYZ, R, f):
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
    org_data, base_data = read_dat_file('..\\data\\lor50.dat')
    Xs, Ys, Zs, phi, omega, kappa, R, control_img_coords, f = solve_orientation(base_data, org_data)
    pixel_residuals = compute_pixel_residuals(base_data[:, 1:4], control_img_coords, [Xs, Ys, Zs], R, f)

    print("Xs:", Xs, "Ys:", Ys, "Zs:", Zs)
    print("phi:", phi, "omega:", omega, "kappa:", kappa)
    print("角锥体像素残差：", pixel_residuals)


if __name__ == "__main__":
    main()
