import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.optimize import fsolve, newton_krylov, broyden1, leastsq
from scipy.optimize import approx_fprime
from scipy.integrate import solve_ivp



def psi_function(row_1, row_2):
    psi_i = row_1[0]
    psi_j = row_2[0]
    if psi_i + psi_j == 0:
        return 0
    return (psi_i - psi_j) *np.abs(psi_i - psi_j)/(psi_i + psi_j)


class Simulator:
    def __init__(self, p, tstep = 0.1) -> None:
        """ ANY changes here must also go to reset method below"""
        self.secant_used = False
        self.t = 0
        self.tstep = tstep
        self.p = p
        # calculate T
        self.N = self.p['coords'].shape[0]
        self._calculate_T()
        self.u = p['u0']
        self.q = p['q0']
        self.num_psis = p['mu'].shape[1]
        # init x with only the S population
        self.x = np.zeros([self.N,self.num_psis])
        self.x[:,0] = self.p['S0']
        self.dead_tracker = [np.zeros([self.N])]
        self.source_sum = np.zeros(self.N)

        # one time calc
        self.T_expanded = np.tile(self.p['T'][...,np.newaxis], [1,1,self.num_psis])
        self.P_expanded = np.tile(self.p['P'][...,np.newaxis], [1,1,self.num_psis])
        self.M_expanded = np.tile(self.p['M'][np.newaxis,...],[self.N,1,1])

    def reset(self, p = None, tstep = None):
        self.t = 0
        self.tstep = tstep
        self.p = p
        # calculate T
        self.N = self.p['coords'].shape[0]
        self._calculate_T()
        self.u = p['u0']
        self.q = p['q0']
        self.num_psis = p['mu'].shape[1]
        self.x = np.zeros([self.N, self.num_psis])
        # init x with only the S population
        self.x = np.zeros([self.N,self.num_psis])
        self.x[:,0] = self.p['S0']
        self.dead_tracker = [np.zeros([self.N])]

        # one time calc
        self.T_expanded = np.tile(self.p['T'][...,np.newaxis], [1,1,self.num_psis])
        self.P_expanded = np.tile(self.p['P'][...,np.newaxis], [1,1,self.num_psis])
        self.M_expanded = np.tile(self.p['M'][np.newaxis,...],[self.N,1,1])

    def _calculate_T(self):
        # T is an N by N matrix
        coords = self.p['coords'] # N x 2
        deltas = pairwise_distances(coords)
        exp_deltas = np.exp(-1 * deltas)
        self.p['T'] = exp_deltas / ((np.sum(exp_deltas) - self.N)/2)
        np.fill_diagonal(self.p['T'], 0)


    def _calculate_tau_matrix(self):
        # tau is N x 4 (SIRD)
        # self.x.shape is N x 4
        psi_matrix = np.array([pairwise_distances(self.x[:,i:i+1], metric = psi_function) for i in range(self.num_psis)])
        psi_matrix = np.transpose(psi_matrix, (1, 2, 0))
        for i in range(self.num_psis):
          psi_matrix[(*np.tril_indices(self.N, -1),i)] *= -1
        result = psi_matrix * self.T_expanded * self.P_expanded * self.M_expanded
        # result = np.floor(psi_matrix * self.T_expanded * self.P_expanded * self.M_expanded)
        return result

    def u_sources(self):
        if self.p['mode'] == 'exp':
            self.u = np.apply_along_axis(lambda x :np.floor(x[0]*np.exp(-1*x[1]*self.t)) , 1, np.vstack([self.p['u0'], self.p['gamma']]).T)
        elif self.p['mode'] == 'lin':
            self.u = np.apply_along_axis(lambda x : np.max([0,np.floor(x[0] - x[1]*self.t)]) , 1, np.vstack([self.p['u0'], self.p['zeta']]).T)

    def q_intra_sources(self):
        if self.p['mode'] == 'exp':
            self.q = np.apply_along_axis(lambda x :np.floor(x[0]*np.exp(-1*x[1]*self.t)) , 1, np.vstack([self.p['q0'], self.p['gamma']]).T)
        elif self.p['mode'] == 'lin':
            self.q = np.apply_along_axis(lambda x : np.max([0,np.floor(x[0] - x[1]*self.t)]) , 1, np.vstack([self.p['q0'], self.p['zeta']]).T)

    def eval_f(self, custom_x = None, active_step = False, dt = None):
        # extract params
        if custom_x is None:
          x = self.x
        else:
          x = custom_x

        if active_step:
            assert dt is not None

        if x.shape[0] == x.size:
            x = np.reshape(x, (self.N, self.num_psis))
        p = self.p
        u = self.u
        q = self.q
        N = p['N']
        mu = p['mu']
        num_psis = self.num_psis
        v = p['v']
        alpha = p['alpha']
        beta = p['beta']
        kappa = p['kappa']
        # init f matrix
        f = np.zeros([N,num_psis])
        # get tau
        tau = self._calculate_tau_matrix()
        # sum tau over AXIS = 1
        tau = np.sum(tau, axis=1) # N x num_psi(4)

        q = np.apply_along_axis(lambda x : x[2] if x[0] - x[1] > x[2] else x[0], 1, np.vstack([q,u, x[:,0]]).T)

        # number of people per node
        sigma = np.sum(x, axis = 1)# + self.dead_tracker[-1]
        # ensure no sigma is zero
        sigma[np.where(sigma == 0)] = 1e-5
        mu[np.where(mu == 0)] = 1e-5
        # evaluate f

        vaccination_rate = (v * (x[:,0]**2)/ (sigma))
        max_vaccination_rate = np.inf  # Set this to a reasonable value
        vaccination_rate = np.minimum(vaccination_rate, max_vaccination_rate)

        # POSSIBLE FIX: do Guarding of u - q AFTER calculating the other terms
        f[:,0] = (-alpha * x[:,1] * x[:,0] / sigma) - mu[:,0] * x[:,0] - q - tau[:,0] - vaccination_rate + u
        f[:,1] = (alpha * x[:,1] * x[:,0] / sigma) - (mu[:,1] + beta + kappa) * x[:,1] + q - tau[:,1]
        f[:,2] = beta * x[:,1] - mu[:,2] * x[:,2] +vaccination_rate - tau[:,2]

        if active_step:
            self.dead_tracker.append(self.dead_tracker[-1] + dt * (kappa * x[:,1] + mu[:,0] * x[:,0] + mu[:,1] * x[:,1]  + mu[:,2] * x[:,2]))
            self.source_sum += dt * u
            self.u_sources()
            self.q_intra_sources()

        return f
    
    def solve_steady_state(self, guess = None):
        if guess is None:
            guess = np.reshape(self.x, (self.N * self.num_psis))
        else:
            guess = np.reshape(guess, (self.N * self.num_psis))
        root = fsolve(self.eval_f, guess, True, xtol=1)
        return root
    
    def Jac_test(self): 
        def _eval_f_wrapper(x_flat):
            x_reshaped = x_flat.reshape(self.x.shape)
            f_val = self.eval_f(x_reshaped)
            return f_val.flatten()

        x = self.x.flatten()
        epsilon = np.sqrt(np.finfo(float).eps)
        jacobian = approx_fprime(x, _eval_f_wrapper, epsilon)
        return jacobian.reshape(self.x.shape[0] * self.x.shape[1], -1)


    def FiniteDifference_JacobianMatrix(self, x_custom=None):
        if x_custom is None:
            x_col = self.x.flatten().reshape(-1, 1)
        else:
            x_col = x_custom.flatten().reshape(-1, 1)

        total_states = len(x_col)
        Jf = np.zeros((total_states, total_states))

        # Adaptive epsilon based on the norm of x_col
        norm_x = np.linalg.norm(x_col, np.inf)
        base_epsilon = 2 * np.sqrt(np.finfo(float).eps) * np.sqrt(1 + norm_x)

        for i in range(total_states):
            epsilon_i = base_epsilon * (1 + abs(x_col[i, 0]))
            e_i = np.zeros((total_states, 1))
            e_i[i, 0] = epsilon_i

            x_perturbed = (x_col + e_i).reshape(self.x.shape)
            delta_f = (self.eval_f(x_perturbed) - self.eval_f(x_perturbed - e_i.reshape(self.x.shape))).flatten()

            Jf[:, i] = delta_f / epsilon_i

        return Jf

    def forward_euler(self, tspan, dt, return_tau = False):
        """
        Uses Forward Euler to simulate states model dx/dt=f(x,p,u)
        starting from state vector x_start at time t_start
        until time t_stop, with time intervals timestep.

        Parameters:
        - f: function that evaluates f(x,p,u)
        - x: initial state vector
        - tspan: tuple of (start, end) time
        - dt: time step for the integration

        Returns:
        - t: array of time points
        - y: array of state vectors at each time point
        """

        t_start = tspan[0]
        t_stop = tspan[1]

        num_steps = len(np.arange(t_start, t_stop, dt))
        t = np.zeros(num_steps + 1)

        # Initialize the state trajectory array
        y = np.zeros((num_steps + 1,) + self.x.shape)
        y[0] = self.x

        self.t = t_start
        t[0] = self.t

        taus = []

        for n in range(num_steps):
            #if np.any(self.x <0):
              #break
            dt = min(dt, (t_stop - t[n]))
            f_val = self.eval_f(dt=dt, active_step=True)
            self.x += dt * f_val
            y[n+1] = self.x
            self.t += dt
            t[n+1] = self.t
            # print(f"Time step {t[n+1]}: Sum of x = {np.sum(self.x)}")
            #print(f"Time step {t[n+1]}: Sum of x = {np.sum(self.x) + np.sum(self.dead_tracker[-1])}")
            # print(f"Time step {t[n+1]}: New Source Sum (per node/total) = {self.source_sum}, {np.sum(self.source_sum)}")
            # if self.t % 10 == 0:
            #     print("Time: ", self.t)
            if return_tau:
              tau = self._calculate_tau_matrix()
              taus.append(tau)
        if return_tau:
          return t, y, taus

        return t, y

    def trapezoidal_equation(self, x_new_flat, x_old_flat, dt):
        """
        Calculate the trapezoidal rule result.

        Parameters:
        - x_new_flat: flattened state at the next time step
        - x_old_flat: flattened state at the current time step
        - dt: time step size

        Returns:
        - result: result of the trapezoidal rule
        """
        # Reshape the flattened states back to their original multi-dimensional structure
        x_new = x_new_flat.reshape(self.x.shape)
        x_old = x_old_flat.reshape(self.x.shape)

        # Calculate f for current and next time steps
        f_current = self.eval_f(custom_x=x_old).flatten()  # Flatten the result
        f_next = self.eval_f(custom_x=x_new).flatten()  # Flatten the result

        # Compute the trapezoidal rule result
        result = x_new_flat - x_old_flat - (dt / 2) * (f_current + f_next)
        return result

    from scipy.optimize import approx_fprime

########################## GCR CODE ####################################

    def trapezoidal_method_GCRHadpt(self, tspan, initial_dt, tolerance, max_dt=1, max_iters_newton=50):
        t_start, t_stop = tspan
        max_estimated_steps = int((t_stop - t_start) / initial_dt) + 1
        t = np.zeros(max_estimated_steps)
        y = np.zeros((max_estimated_steps,) + self.x.shape)
        y[0] = self.x.copy()
        t[0] = t_start

        self.t = t_start
        dt = initial_dt
        n = 0

        while self.t < t_stop and n < max_estimated_steps - 1:
            x_current = self.x.copy()
            f_current = self.eval_f(x_current, active_step=True, dt=dt)

            def trapezoidal_implicit_function(x_next_flat):
                x_next = x_next_flat.reshape(self.x.shape)
                f_next = self.eval_f(x_next)
                return (x_next - x_current - (dt / 2) * (f_current + f_next)).flatten()

            x_next_flat_guess = x_current.flatten()
            x_next_flat, converged = self.HybridSolver_GCR(
                x_next_flat_guess,
                trapezoidal_implicit_function,
                max_iters_newton = max_iters_newton
            )

            if not converged:
                print(f"Newton-GCR method failed to converge at time step {n + 1}. Decreasing dt.")
                dt = max(initial_dt, dt / 1.5)
                continue

            self.x = x_next_flat.reshape(self.x.shape)
            y[n + 1] = self.x.copy()

            # Calculate time remaining and adjust dt
            time_remaining = t_stop - self.t
            dt = min(dt, time_remaining)  # Ensure dt does not exceed time remaining

            self.t += dt
            t[n + 1] = self.t
            n += 1

            
            change = np.linalg.norm(self.x - x_current, np.inf)
            gradual_reduction_threshold = 0 * (t_stop - t_start)  

            if self.t >= t_stop - gradual_reduction_threshold:
                dt = max(initial_dt, dt / 1.1)  
            elif change < tolerance / 2 and dt < max_dt:
                dt = min(dt * 1.1, max_dt, time_remaining)  
            #elif change > tolerance:
                #dt = max(initial_dt, dt / 1.5)  # Decrease dt if change is large

            print(f"Time step {n + 1}: dt = {dt}, change = {change}")

        t = t[:n + 1]
        y = y[:n + 1]

        return t, y
    

    def trapezoidal_method_GCRadpt(self, tspan, initial_dt, tolerance, max_dt=1, max_iters_newton=50):
        t_start, t_stop = tspan
        max_estimated_steps = int((t_stop - t_start) / initial_dt) + 1
        t = np.zeros(max_estimated_steps)
        y = np.zeros((max_estimated_steps,) + self.x.shape)
        y[0] = self.x.copy()
        t[0] = t_start

        self.t = t_start
        dt = initial_dt  # Start with the minimum dt
        n = 0

        while self.t < t_stop and n < max_estimated_steps - 1:
            x_current = self.x.copy()
            f_current = self.eval_f(x_current, active_step=True, dt=dt)

            def trapezoidal_implicit_function(x_next_flat):
                x_next = x_next_flat.reshape(self.x.shape)
                f_next = self.eval_f(x_next)
                return (x_next - x_current - (dt / 2) * (f_current + f_next)).flatten()

            x_next_flat_guess = x_current.flatten()
            x_next_flat, converged = self.NewtonNd_GCR(
                x_next_flat_guess,
                trapezoidal_implicit_function,
                tol_f=tolerance,
                max_iter=max_iters_newton
            )

            if not converged:
                print(f"Newton-GCR method failed to converge at time step {n + 1}. Decreasing dt.")
                dt = max(initial_dt, dt / 1.1)
                continue

            self.x = x_next_flat.reshape(self.x.shape)
            y[n + 1] = self.x.copy()

            time_remaining = t_stop - self.t
            dt = min(dt, time_remaining)  

            self.t += dt
            t[n + 1] = self.t
            n += 1

            # Adjust dt based on solution change and phase of simulation
            change = np.linalg.norm(self.x - x_current, np.inf)
            gradual_reduction_threshold = 0.005 * (t_stop - t_start)  

            if self.t >= t_stop - gradual_reduction_threshold:
                dt = max(initial_dt, dt / 1.25)  
            elif change < tolerance / 2 and dt < max_dt:
                dt = min(dt * 1.25, max_dt, time_remaining)  
            #elif change > tolerance:
                #dt = max(initial_dt, dt / 1.5)  # Decrease dt if change is large

            print(f"Time step {n + 1}: dt = {dt}, change = {change}")

        # Trim the arrays to the actual number of steps taken
        t = t[:n + 1]
        y = y[:n + 1]

        return t, y

    def trapezoidal_method_GCR(self, tspan, dt):
            t_start, t_stop = tspan
            num_steps = int((t_stop - t_start) / dt)
            t = np.zeros(num_steps + 1)
            y = np.zeros((num_steps + 1,) + self.x.shape)
            y[0] = self.x

            self.t = t_start
            t[0] = self.t
            for n in range(num_steps):
                x_current = self.x
                f_current = self.eval_f(x_current, active_step=True, dt=dt)
                def trapezoidal_implicit_function(x_next_flat, f_current):
                    x_next = x_next_flat.reshape(self.x.shape)
                    f_next = self.eval_f(x_next)
                    return (x_next - x_current - (dt / 2) * (f_current + f_next)).flatten()

                x_next_flat_guess = x_current.flatten()
                # Use newtonNd_GCR for solving the implicit equation
                x_next_flat, converged= self.NewtonNd_GCR(
                    x_next_flat_guess,
                    trapezoidal_implicit_function,
                    tol_f=1e-9,  
                    max_iter=20,  
                )

                if not converged:
                    print(f"Newton-GCR method failed to converge at time step {n + 1}")

                self.x = x_next_flat.reshape(self.x.shape)
                y[n + 1] = self.x

                self.t += dt
                t[n + 1] = self.t

            return t, y

    def tgcr_matrixfree(self, xk, b, eps, tolrGCR, MaxItersGCR):
            x = np.zeros_like(b)
            r = b.copy()
            r_norms = [np.linalg.norm(r, 2)]
            P_matrix = np.zeros((len(b), MaxItersGCR))
            gcr_converged = False  

            for k in range(MaxItersGCR):
                P_matrix[:, k] = r / (np.linalg.norm(r) + 1e-15)  
                Ap = (self.eval_f(xk + eps * P_matrix[:, k]) - self.eval_f(xk)) / eps 
                Ap = Ap.flatten()

                # Orthogonalize Ap against previous directions in P_matrix
                for j in range(k):
                    Ap -= np.dot(Ap, P_matrix[:, j]) * P_matrix[:, j]

                Ap_norm = np.linalg.norm(Ap)
                if Ap_norm < 1e-15:  
                    break

                Ap /= Ap_norm
                P_matrix[:, k] = Ap

                
                alpha = np.dot(r, Ap)
                x += alpha * Ap
                r -= alpha * Ap
                new_r_norm = np.linalg.norm(r)
                r_norms.append(new_r_norm)

                
                if new_r_norm <= tolrGCR * r_norms[0]:
                    gcr_converged = True
                    break

            return x, gcr_converged, r_norms


    def NewtonNd_GCR(self, x0_flat, eval_f, tol_f=1e-9, max_iter=1000, custom_J = None, return_res = False):
            k = 0
            x_flat = x0_flat
            f = eval_f(x_flat)
            err_f = np.linalg.norm(f, np.inf)
            tolrGCR = 1e-8
            MaxItersGCR = 10000
            eps = 1e-4; 
            eps_a = np.sqrt(np.finfo(float).eps * eps)

            while k < max_iter:

                delta_x, _, _ = self.tgcr_matrixfree(x_flat, -f.flatten(), eps_a, tolrGCR, MaxItersGCR)
                
                step_size = 1 # Adjust based on your problem scale

                x_new_flat = x_flat + step_size * delta_x
                x_flat = x_new_flat
                k += 1

                f = eval_f(x_flat)
                err_f = np.linalg.norm(f, np.inf)

                #print(f"Iteration {k}: Residual norm = {err_f}")

                if err_f <= tol_f:
                    #print(f"Newton's method converged in {k} iterations with residual norm: {err_f}")
                    if return_res:
                        return x_flat, True, err_f
                    else:
                        return x_flat, True

            print("Newton did NOT converge! Maximum Number of Iterations reached")
            if return_res:
                return x_flat, True, err_f
            else:
                return x_flat, True
        
    def HybridSolver_GCR(self, x0_flat, func, max_iters_newton):
        if not self.secant_used:
            # Use the Secant method only if it hasn't been used before
            x1_flat = x0_flat + np.random.randn(*x0_flat.shape) * 1e-4
            x_initial_guess, secant_converged = self.secant_method_solver(x0_flat, x1_flat, func)
            self.secant_used = True
        else:
            # If Secant has been used before, just use the current guess
            x_initial_guess = x0_flat

        # Proceed with Newton-GCR method
        x_newton_GCR, newton_GCR_converged, final_residual_norm = self.NewtonNd_GCR(
            x_initial_guess,
            func,
            tol_f=1e-9,
            max_iter=max_iters_newton,
            return_res=True
        )

        return x_newton_GCR, newton_GCR_converged

################### NEWTON NORMAL TRAP CODE ####################################

    def trapezoidal_method(self, tspan, dt):
        t_start, t_stop = tspan
        num_steps = int((t_stop - t_start) / dt)
        t = np.zeros(num_steps + 1)
        y = np.zeros((num_steps + 1,) + self.x.shape)
        y[0] = self.x

        self.t = t_start
        t[0] = self.t

        for n in range(num_steps):
            x_current = self.x
            f_current = self.eval_f(x_current, active_step=True, dt=dt)

            def trapezoidal_implicit_function(x_next_flat):
                x_next = x_next_flat.reshape(self.x.shape)
                f_next = self.eval_f(x_next)
                return (x_next - x_current - (dt / 2) * (f_current + f_next)).flatten()
            
            def J_trapezoidal_implicit_function(x_next_flat):
                x_next = x_next_flat.reshape(self.x.shape)
                return np.eye(x_next.size) - dt/2 * self.FiniteDifference_JacobianMatrix(x_next)        

            x_next_flat_guess = x_current.flatten()
            x_next_flat, converged = self.NewtonNd(x_next_flat_guess, trapezoidal_implicit_function,
                                                   custom_J = J_trapezoidal_implicit_function)

            if not converged:
                print(f"Newton's method failed to converge at time step {n + 1}")

            self.x = x_next_flat.reshape(self.x.shape)
            y[n + 1] = self.x

            self.t += dt
            t[n + 1] = self.t

        return t, y

    def NewtonNd(self, x0_flat, eval_f, tol_f=1e-4, max_iter=1000, custom_J = None, return_res = False):
        k = 0
        x_flat = x0_flat
        f = eval_f(x_flat)
        err_f = np.linalg.norm(f, np.inf)

        while k < max_iter:
            if custom_J is None:
                Jf = self.FiniteDifference_JacobianMatrix(x_flat)
            else:
                Jf = custom_J(x_flat)

            delta_x = np.linalg.solve(Jf, -f.flatten())

            # Fixed step size
            step_size = 1 # Adjust based on your problem scale

            x_new_flat = x_flat + step_size * delta_x
            x_flat = x_new_flat
            k += 1

            f = eval_f(x_flat)
            err_f = np.linalg.norm(f, np.inf)

            #print(f"Iteration {k}: Residual norm = {err_f}")

            if err_f <= tol_f:
                #print(f"Newton's method converged in {k} iterations with residual norm: {err_f}")
                if return_res:
                    return x_flat, True, err_f
                else:
                    return x_flat, True

        print("Newton did NOT converge! Maximum Number of Iterations reached")
        if return_res:
            return x_flat, True, err_f
        else:
            return x_flat, True

    def secant_method_solver(self, x0_flat, x1_flat, func, epsilon=1e-2, max_iters=1000):
        for _ in range(max_iters):
            f_x0 = func(x0_flat)
            f_x1 = func(x1_flat)

            if np.linalg.norm(f_x1 - f_x0) < 1e-8:
                print("Denominator too small. Secant method failed.")
                return x1_flat, False

            x2_flat = x1_flat - f_x1 * (x1_flat - x0_flat) / (f_x1 - f_x0)
            if np.linalg.norm(x2_flat - x1_flat) < epsilon:
                return x2_flat, True  # Converged

            x0_flat, x1_flat = x1_flat, x2_flat

        print("Secant method did not converge.")
        return x1_flat, False

    def HybridSolver_s(self, x0_flat, func, J_func, convergence_threshold, max_iters_newton):
        # Always use the Secant method to generate an initial guess
        x1_flat = x0_flat + np.random.randn(*x0_flat.shape) * 1e-2
        x_initial_guess, secant_converged = self.secant_method_solver(x0_flat, x1_flat, func)

        # Proceed with Newton's method using the initial guess from the Secant method
        x_newton, newton_converged, final_residual_norm = self.NewtonNd(x_initial_guess, func, max_iters_newton, custom_J=J_func, return_res=True)

        return x_newton, newton_converged


    def HybridSolver(self, x0_flat, func, J_func, convergence_threshold, max_iters_newton):
        if not self.secant_used:
            # Use the Secant method only if it hasn't been used before
            x1_flat = x0_flat + np.random.randn(*x0_flat.shape) * 1e-4
            x_initial_guess, secant_converged = self.secant_method_solver(x0_flat, x1_flat, func)
            self.secant_used = True  
            #print("Secant method used for initial guess.")
        else:
            # If Secant has been used before, just use the current guess
            x_initial_guess = x0_flat
            #print("Using current guess for Newton's method.")

        # Proceed with Newton's method
        x_newton, newton_converged, final_residual_norm = self.NewtonNd(x_initial_guess, func, max_iters_newton, custom_J=J_func, return_res = True)

        #if newton_converged:
            #print(f"Newton's method converged with final residual norm: {final_residual_norm}")
        #else:
            #print("Newton's method did not converge.")

        return x_newton, newton_converged
    
    def trapezoidal_method_sec(self, tspan, dt):
        t_start, t_stop = tspan
        num_steps = int((t_stop - t_start) / dt)
        t = np.zeros(num_steps + 1)
        y = np.zeros((num_steps + 1,) + self.x.shape)
        y[0] = self.x

        self.t = t_start
        t[0] = self.t

        for n in range(num_steps):
            x_current = self.x
            f_current = self.eval_f(x_current, active_step=True, dt=dt)

            def trapezoidal_implicit_function(x_next_flat):
                x_next = x_next_flat.reshape(self.x.shape)
                f_next = self.eval_f(x_next)
                return (x_next - x_current - (dt / 2) * (f_current + f_next)).flatten()
            

            x_next_flat_guess = x_current.flatten()
            x_next_flat_guess_prev = x_next_flat_guess + 1e-4  # Slightly perturbed guess

            x_next_flat, converged = self.secant_method_solver(
                x_next_flat_guess_prev, x_next_flat_guess, trapezoidal_implicit_function)

            if not converged:
                print(f"Secant method failed to converge at time step {n + 1}")

            self.x = x_next_flat.reshape(self.x.shape)
            y[n + 1] = self.x

            self.t += dt
            t[n + 1] = self.t

        return t, y
    
    def trapezoidal_method_hybrid(self, tspan, dt, convergence_threshold=1e-1, max_iters_newton=50):
        t_start, t_stop = tspan
        num_steps = int((t_stop - t_start) / dt)
        t = np.zeros(num_steps + 1)
        y = np.zeros((num_steps + 1,) + self.x.shape)
        y[0] = self.x

        self.t = t_start
        t[0] = self.t

        for n in range(num_steps):
            x_current = self.x
            f_current = self.eval_f(x_current, active_step=True, dt=dt)

            def trapezoidal_implicit_function(x_next_flat):
                x_next = x_next_flat.reshape(self.x.shape)
                f_next = self.eval_f(x_next)
                return (x_next - x_current - (dt / 2) * (f_current + f_next)).flatten()

            def J_trapezoidal_implicit_function(x_next_flat):
                x_next = x_next_flat.reshape(self.x.shape)
                return np.eye(x_next.size) - dt/2 * self.FiniteDifference_JacobianMatrix(x_next)   

            x_next_flat_guess = x_current.flatten()

            # Using Hybrid Solver
            x_next_flat, converged = self.HybridSolver(x_next_flat_guess, trapezoidal_implicit_function, J_trapezoidal_implicit_function, convergence_threshold, max_iters_newton)

            if not converged:
                print(f"Solver failed to converge at time step {n + 1}")

            self.x = x_next_flat.reshape(self.x.shape)
            y[n + 1] = self.x

            self.t += dt
            t[n + 1] = self.t

        return t, y
    
    def trapezoidal_method_adapt(self, tspan, initial_dt, tolerance, convergence_threshold=1e-1, max_dt=1, max_iters_newton=50):
        t_start, t_stop = tspan
        max_estimated_steps = int((t_stop - t_start) / initial_dt) + 1  # Estimate based on initial_dt for maximum steps
        t = np.zeros(max_estimated_steps)
        y = np.zeros((max_estimated_steps,) + self.x.shape)
        y[0] = self.x.copy()
        t[0] = t_start

        self.t = t_start
        dt = initial_dt  # Start with initial_dt
        n = 0

        while self.t < t_stop and n < max_estimated_steps - 1:
            x_current = self.x.copy()
            f_current = self.eval_f(x_current, active_step=True, dt=dt)
            
            def trapezoidal_implicit_function(x_next_flat):
                x_next = x_next_flat.reshape(self.x.shape)
                f_next = self.eval_f(x_next)
                return (x_next - x_current - (dt / 2) * (f_current + f_next)).flatten()

            def J_trapezoidal_implicit_function(x_next_flat):
                x_next = x_next_flat.reshape(self.x.shape)
                return np.eye(x_next.size) - dt/2 * self.FiniteDifference_JacobianMatrix(x_next)

            x_next_flat_guess = x_current.flatten()
            x_next_flat, converged = self.HybridSolver(x_next_flat_guess, trapezoidal_implicit_function, J_trapezoidal_implicit_function, convergence_threshold, max_iters_newton)

            if not converged:
                print(f"Newton-GCR method failed to converge at time step {n + 1}. Decreasing dt.")
                dt = max(initial_dt, dt / 1.1)
                continue

            self.x = x_next_flat.reshape(self.x.shape)
            y[n + 1] = self.x.copy()

            # Calculate time remaining and adjust dt
            time_remaining = t_stop - self.t
            dt = min(dt, time_remaining)  # Ensure dt does not exceed time remaining

            self.t += dt
            t[n + 1] = self.t
            n += 1

            # Adjust dt based on solution change and phase of simulation
            change = np.linalg.norm(self.x - x_current, np.inf)
            gradual_reduction_threshold = 0.025 * (t_stop - t_start)  

            if self.t >= t_stop - gradual_reduction_threshold:
                dt = max(initial_dt, dt / 1.1)  
            elif change < tolerance / 2 and dt < max_dt:
                dt = min(dt * 1.1, max_dt, time_remaining)  
            #elif change > tolerance:
                #dt = max(initial_dt, dt / 1.5)  # Decrease dt if change is large

            print(f"Time step {n + 1}: dt = {dt}, change = {change}")

        # Trim the arrays to the actual number of steps taken
        t = t[:n + 1]
        y = y[:n + 1]

        return t, y

    # Wrapped functions
    def wrapped_eval_f(self, x_flat):
        x_reshaped = x_flat.reshape((self.N, self.num_psis))
        f_val = self.eval_f(x_reshaped)
        return f_val.flatten()

    def wrapped_J(self, x_flat):
        x_reshaped = x_flat.reshape((self.N, self.num_psis))
        J = self.FiniteDifference_JacobianMatrix(x_reshaped)
        return J

    def backward_euler(self, tspan, dt):
        t_start, t_stop = tspan
        num_steps = int((t_stop - t_start) / dt)
        t = np.zeros(num_steps + 1)
        y = np.zeros((num_steps + 1,) + self.x.shape)
        y[0] = self.x

        self.t = t_start
        t[0] = self.t

        epsilon_f = 1e-1  # Tolerance for function convergence
        epsilon_deltax = 1e-1  # Tolerance for change in x
        epsilon_xrel = np.inf  # Relative tolerance for x
        max_iters = 100  # Maximum iterations for Newton's method

        for n in range(num_steps):
            # run active step for dead tracker and sources
            _ = self.eval_f(active_step=True, dt=dt)
            x_next_flat_guess = self.x.flatten()

            def implicit_function(x_next_flat):
                x_next = x_next_flat.reshape(self.x.shape)
                return (x_next - self.x - dt * self.eval_f(x_next)).flatten()
            
            def J_backward_implicit_function(x_next_flat):
                x_next = x_next_flat.reshape(self.x.shape)
                return np.eye(x_next.size) - dt* self.FiniteDifference_JacobianMatrix(x_next) 

            # Extract only the first element of the tuple (the state x) from the solver
            x_next_flat, converged = self.NewtonNd(x_next_flat_guess, implicit_function,
                                                   custom_J = J_backward_implicit_function)

            self.x = x_next_flat.reshape(self.x.shape)
            print(f"Time step {n + 1}: Sum of x = {np.sum(self.x)}")
            y[n + 1] = self.x
            self.t += dt
            t[n + 1] = self.t
            if self.t % 10 == 0:
                print("Time: ", self.t)

        return t, y
    

    def backward_euler_f(self, tspan, dt):
        t_start, t_stop = tspan
        num_steps = int((t_stop - t_start) / dt)
        t = np.zeros(num_steps + 1)
        y = np.zeros((num_steps + 1,) + self.x.shape)
        y[0] = self.x

        self.t = t_start
        t[0] = self.t

        for n in range(num_steps):
            # run active step for dead tracker and sources
            _ = self.eval_f(active_step=True, dt=dt)
            x_next_flat_guess = self.x.flatten()

            def implicit_function(x_next_flat):
                x_next = x_next_flat.reshape(self.x.shape)
                return (x_next - self.x - dt * self.eval_f(x_next)).flatten()

            # Solve using fsolve
            x_next_flat = fsolve(implicit_function, x_next_flat_guess)

            self.x = x_next_flat.reshape(self.x.shape)
            print(f"Time step {n + 1}: Sum of x = {np.sum(self.x)}")
            y[n + 1] = self.x
            self.t += dt
            t[n + 1] = self.t
            if self.t % 10 == 0:
                print("Time: ", self.t)

        return t, y

    def trapezoidal_method_f(self, tspan, dt):
        t_start, t_stop = tspan
        num_steps = int((t_stop - t_start) / dt)
        t = np.zeros(num_steps + 1)
        y = np.zeros((num_steps + 1,) + self.x.shape)
        y[0] = self.x

        self.t = t_start
        t[0] = self.t

        for n in range(num_steps):
            x_current = self.x
            f_current = self.eval_f(x_current, active_step=True, dt=dt)

            def trapezoidal_implicit_function(x_next_flat):
                x_next = x_next_flat.reshape(self.x.shape)
                f_next = self.eval_f(x_next)
                res = x_next - x_current - (dt / 2) * (f_current + f_next)
                return res.flatten()

            x_next_flat_guess = x_current.flatten()
            x_next_flat = fsolve(trapezoidal_implicit_function, x_next_flat_guess)
            print(f"Time step {n + 1}: before reshaping  = {x_next_flat}")
            self.x = x_next_flat.reshape(self.x.shape)
            y[n + 1] = self.x

            self.t += dt
            t[n + 1] = self.t
            if self.t % 10 == 0:
                print("Time: ", self.t)

        return t, y