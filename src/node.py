import numpy as np
import config


class maxcut:
    def __init__(self, Q, id=1, level=0, sdp_mult=None, tri=None, tri2=None, get_ub=lambda: None, scale=1, Xfrac=None, vertex_map=None, vertex_sign=None):

        self.id = id
        self.level = level

        self.n = Q.shape[0]

        self.conf = self.get_conf()

        self.Q = Q
        self.C = np.empty_like(Q)
        self._QQT = np.empty_like(Q)

        self.X = np.empty_like(Q)

        self.Xfrac = np.copy(Xfrac) if Xfrac is not None else None

        self.scale = scale

        self.bound_pruned = False
        self.n_fixed = 0

        self.ub = None
        self.lb = None

        self.external_ub = get_ub

        n = self.n
        if self.conf['projected']:
            self.sdp_mult = np.copy(sdp_mult) if sdp_mult is not None else np.zeros((n,n))
        else:
            self.sdp_mult = np.copy(sdp_mult) if sdp_mult is not None else np.eye(n)

        self.tri_mult = np.array(tri['mult']) if tri is not None else []
        self.tri = tri['tri'].copy() if tri is not None else []

        self.tri2_mult = np.array(tri2['mult']) if tri2 is not None else []
        self.tri2 = tri2['tri'].copy() if tri2 is not None else []

        self.recalculate_indices_tri()
        self.recalculate_indices_tri2()

        self.fixed_mask = np.zeros((n, n), dtype=bool)
        self.fixed_val_mat = np.zeros((n, n), dtype=Q.dtype)
        self.candidate_mask = np.triu(np.ones((n, n), dtype=bool), k=1)

        self.vertex_map  = np.copy(vertex_map)  if vertex_map  is not None else np.arange(n)
        self.vertex_sign = np.copy(vertex_sign) if vertex_sign is not None else np.ones(n, dtype=np.int8)


    def recalculate_indices_tri(self):
        if len(self.tri) == 0: return

        L_arr = np.asarray(self.tri, dtype=np.intp)
        self.IJ_tri = L_arr[:, 0] * self.X.shape[1] + L_arr[:, 1]
        self.IK_tri = L_arr[:, 0] * self.X.shape[1] + L_arr[:, 2]
        self.JK_tri = L_arr[:, 1] * self.X.shape[1] + L_arr[:, 2]

    def recalculate_indices_tri2(self):
        if len(self.tri2) == 0: return

        L_arr = np.asarray(self.tri2, dtype=np.intp)
        self.IJ_tri2 = L_arr[:, 0] * self.X.shape[1] + L_arr[:, 2]
        self.IK_tri2 = L_arr[:, 0] * self.X.shape[1] + L_arr[:, 1]
        self.JK_tri2 = L_arr[:, 1] * self.X.shape[1] + L_arr[:, 2]

    def get_ub(self):
        ext = self.external_ub()
        if self.ub is None and ext is None:
            return None, None
        if self.ub is None:
            return ext
        if ext is None or ext[0] is None:
            return self.ub, self.x
        if self.ub < ext[0]:
            return self.ub, self.x
        return ext

    def get_conf(self):
        return config.get_conf(self.id == 1)

    def get_modified_costs(self):
        np.copyto(self.C, self.Q)
        if self.conf['projected']:
            self.C -= self.scale*self.sdp_mult 
        else:
            self.C -= self.scale*self.sdp_mult@self.sdp_mult.T

        if len(self.tri) > 0:
            tri_mult = -self.scale * self.tri_mult
            Cflat = self.C.ravel()
            np.add.at(Cflat, self.IJ_tri, tri_mult)
            np.add.at(Cflat, self.IK_tri, tri_mult)
            np.add.at(Cflat, self.JK_tri, tri_mult)

        if len(self.tri2) > 0:
            tri_mult = self.scale * self.tri2_mult
            Cflat = self.C.ravel()
            np.add.at(Cflat, self.IJ_tri2, tri_mult)
            np.add.at(Cflat, self.IK_tri2, tri_mult)
            np.add.at(Cflat, self.JK_tri2, -tri_mult)

    def GilmoreLawler(self, Q):
        np.add(Q, Q.T, out=self._QQT)
        np.copyto(self.X, 1.0)
        self.X[self._QQT > 0] = -1
        QQT = self._QQT
        X = self.X

        X[self.fixed_mask] = self.fixed_val_mat[self.fixed_mask]
        X[self.fixed_mask.T] = self.fixed_val_mat.T[self.fixed_mask.T]
        np.fill_diagonal(X, 1)

        obj = np.sum(Q*X) -self.scale*(np.sum(self.tri_mult) if len(self.tri) > 0 else 0) -self.scale*(np.sum(self.tri2_mult) if len(self.tri2) > 0 else 0)

        ub = X[:,0].T@self.Q@X[:,0]

        if self.ub == None or ub < self.ub:
            self.ub = ub
            self.x = self.vertex_sign * X[self.vertex_map, 0]

        delta_thresh = self.get_ub()[0] - obj
        if config.FIXING and delta_thresh > 0:
            deltas = -2*X*QQT
            to_fix = (np.ceil(deltas+obj-1e-5) >= round(self.get_ub()[0])) & self.candidate_mask
            new_fix = to_fix & ~self.fixed_mask
            if np.any(new_fix):
                self.fixed_mask[new_fix] = True
                self.fixed_val_mat[new_fix] = X[new_fix]

        if self.Xfrac is None:
            self.Xfrac = np.copy(self.X)
        else:
            self.Xfrac *= self.conf['alpha']
            self.Xfrac += (1.0-self.conf['alpha'])*self.X


        return obj

    def local_search(self, x, Q):
        x = np.asarray(x, dtype=np.float64).copy()
        n = len(x)
        Q = np.asarray(Q, dtype=np.float64)
        
        if x.shape != (n,) or Q.shape != (n, n):
            raise ValueError("x must be 1D (n,), Q must be (n,n)")
        
        Qx  = Q @ x 
        Qtx = Q.T @ x
        current_cost = float(x @ Qx)
        
        improved = True
        while improved:
            improved = False
            for i in range(n):
                old_xi = x[i]
                
                gain = -2.0 * old_xi * Qtx[i] - 2.0 * old_xi * Qx[i] + 4.0 * Q[i, i]
                
                if gain < 0:  
                    x[i] = -old_xi
                    
                    delta_x = -2.0 * old_xi
                    Qx  += delta_x * Q[:, i] 
                    Qtx += delta_x * Q[i, :]
                    
                    current_cost += gain

                    improved = True
                    break 
        
        return x

    def solve_lag_sub(self):
        self.get_modified_costs()
        return self.GilmoreLawler(self.C)

    def get_semidefiniteness_subgradient_(self):
        X = self.get_X()

        if self.conf['projected']:
            return -X
        return -self.scale*X@self.sdp_mult

    def project(self, M):
        A = (M-M.T)/2
        S = M-A

        C, D = np.linalg.eigh(S)
        C = np.diag( np.maximum(C, 0) )
        P = D@C@D.T

        P += A

        return P

    def get_tri_subgradient(self):
        if len(self.tri) > 0:
            X = self.get_X().ravel()

            s = -X[self.IJ_tri]-X[self.IK_tri]-X[self.JK_tri]-1

            return self.scale*s

        return 0

    def get_tri2_subgradient(self):
        if len(self.tri2) > 0:
            X = self.get_X().ravel()

            s = X[self.IJ_tri2]+X[self.IK_tri2]-X[self.JK_tri2]-1

            return self.scale*s

        return 0

    def update_multipliers(self, k):

        sdp_mult = self.sdp_mult
        sdp_sub = self.get_semidefiniteness_subgradient_()
        norm_sdp = float(np.dot(sdp_sub.ravel(), sdp_sub.ravel()))
        step_sdp = self.conf['get_stepsize'](ub=self.get_ub()[0], lb=self.lb, norm=norm_sdp, k=k)

        sdp_mult += step_sdp*sdp_sub

        if self.conf['projected']:
            self.sdp_mult = self.project(sdp_mult)

        if len(self.tri) > 0:
            tri_mult = self.tri_mult
            tri_sub = self.get_tri_subgradient()
            norm_tri = float(np.dot(tri_sub, tri_sub))
            step_tri = self.conf['get_stepsize'](ub=self.get_ub()[0], lb=self.lb, norm=norm_tri, k=k)
            self.tri_mult += step_tri*tri_sub
            np.maximum(self.tri_mult, 0, out=self.tri_mult)

        if len(self.tri2) > 0:
            tri2_mult = self.tri2_mult
            tri2_sub = self.get_tri2_subgradient()
            norm_tri2 = float(np.dot(tri2_sub, tri2_sub))
            step_tri2 = self.conf['get_stepsize'](ub=self.get_ub()[0], lb=self.lb, norm=norm_tri2, k=k)

            self.tri2_mult += step_tri2*tri2_sub
            np.maximum(self.tri2_mult, 0, out=self.tri2_mult)


    def get_X(self):
        return self.Xfrac if self.conf['frac'] else self.X

    def rem_tri(self):
        if len(self.tri) == 0: return

        rem = np.where(self.tri_mult > 1e-6)[0]

        self.tri = [self.tri[i] for i in rem]
        self.tri_mult = self.tri_mult[rem]
        self.recalculate_indices_tri()

    def rem_tri2(self):
        if len(self.tri2) == 0: return

        rem = np.where(self.tri2_mult > 1e-6)[0]

        self.tri2 = [self.tri2[i] for i in rem]
        self.tri2_mult = self.tri2_mult[rem]
        self.recalculate_indices_tri2()

    def separate_tri(self):
        if len(self.tri) >= self.conf['cuts']['max_tri']:
            return

        X = self.get_X()
        Xs = (X + X.T) / 2
        n = self.n
        threshold = 1e-5

        # vio[i,j,k] = -1 - Xs[i,j] - Xs[i,k] - Xs[j,k]
        vio = -1.0 - Xs[:, :, np.newaxis] - Xs[:, np.newaxis, :] - Xs[np.newaxis, :, :]

        r = np.arange(n)
        upper = (r[:, None, None] < r[None, :, None]) & (r[None, :, None] < r[None, None, :])
        ii, jj, kk = np.where(upper & (vio > threshold))

        if len(ii) == 0:
            return

        order = np.argsort(-vio[ii, jj, kk])
        ii, jj, kk = ii[order], jj[order], kk[order]

        constraints = set(self.tri)
        tri_mult = list(self.tri_mult)
        added = 0
        for idx in range(len(ii)):
            if added >= self.conf['cuts']['max_add'] or len(self.tri) >= self.conf['cuts']['max_tri']:
                break
            t = (int(ii[idx]), int(jj[idx]), int(kk[idx]))
            if t not in constraints:
                self.tri.append(t)
                tri_mult.append(0.0)
                constraints.add(t)
                added += 1

        self.tri_mult = np.array(tri_mult)
        self.recalculate_indices_tri()

    def separate_tri2(self):
        if len(self.tri2) >= self.conf['cuts']['max_tri2']:
            return

        X = self.get_X()
        Xs = (X + X.T) / 2
        n = self.n
        threshold = 1e-5

        # Three violation types for base triple (i,j,k) with i<j<k:
        # vio1[i,j,k] = -1 + Xs[i,j] + Xs[i,k] - Xs[j,k]  → constraint (i,j,k)
        # vio2[i,j,k] = -1 + Xs[j,k] + Xs[i,j] - Xs[i,k]  → constraint (j,i,k)
        # vio3[i,j,k] = -1 + Xs[j,k] + Xs[i,k] - Xs[i,j]  → constraint (k,i,j)
        vio1 = -1.0 + Xs[:, :, np.newaxis] + Xs[:, np.newaxis, :] - Xs[np.newaxis, :, :]
        vio2 = -1.0 + Xs[np.newaxis, :, :] + Xs[:, :, np.newaxis] - Xs[:, np.newaxis, :]
        vio3 = -1.0 + Xs[np.newaxis, :, :] + Xs[:, np.newaxis, :] - Xs[:, :, np.newaxis]

        r = np.arange(n)
        upper = (r[:, None, None] < r[None, :, None]) & (r[None, :, None] < r[None, None, :])

        ii1, jj1, kk1 = np.where(upper & (vio1 > threshold))
        ii2, jj2, kk2 = np.where(upper & (vio2 > threshold))
        ii3, jj3, kk3 = np.where(upper & (vio3 > threshold))

        # Map base-triple indices to constraint keys
        all_ci  = np.concatenate([ii1, jj2, kk3])
        all_cj  = np.concatenate([jj1, ii2, ii3])
        all_ck  = np.concatenate([kk1, kk2, jj3])
        all_vio = np.concatenate([vio1[ii1, jj1, kk1], vio2[ii2, jj2, kk2], vio3[ii3, jj3, kk3]])

        if len(all_vio) == 0:
            return

        order = np.argsort(-all_vio)
        all_ci, all_cj, all_ck = all_ci[order], all_cj[order], all_ck[order]

        constraints = set(self.tri2)
        tri2_mult = list(self.tri2_mult)
        added = 0
        for idx in range(len(all_ci)):
            if added >= self.conf['cuts']['max_add'] or len(self.tri2) >= self.conf['cuts']['max_tri2']:
                break
            t = (int(all_ci[idx]), int(all_cj[idx]), int(all_ck[idx]))
            if t not in constraints:
                self.tri2.append(t)
                tri2_mult.append(0.0)
                constraints.add(t)
                added += 1

        self.tri2_mult = np.array(tri2_mult)
        self.recalculate_indices_tri2()

    def lagrangian_relaxation(self):
        if self.lb == None:
            self.lb = -np.inf

        self.bound_pruned = False

        for k in range(1, self.conf['max_iter']+1):
            lrp = self.solve_lag_sub()

            if lrp > self.lb:
                self.lb = lrp

            if self.get_ub()[0] is not None and np.ceil(self.lb-1e-5) >= round(self.get_ub()[0]):
                self.bound_pruned = True
                break

            if self.conf['cuts']['add'] and k % self.conf['cuts']['freq'] == 0 and k <= self.conf['cuts']['max_iter'] and k > self.conf['cuts']['min_sep']:
                self.separate_tri()
                self.separate_tri2()

            if self.conf['cuts']['rem'] and k % self.conf['cuts']['rem_freq'] == 0:
                self.rem_tri()
                self.rem_tri2()

            if k % 1000 == 0 and self.id == 1:
                x = self.local_search(self.X[0,:], self.Q)
                ub = x.T@self.Q@x
                if self.ub == None or ub < self.ub:
                    self.ub = ub
                    self.x = self.vertex_sign * x[self.vertex_map]

            self.update_multipliers(k)

    def solve_node(self):
        self.lagrangian_relaxation()
        
        self.n_fixed = int(self.fixed_mask.sum())
        
        if not self.bound_pruned:
            self.collapse()

    def _create_child(self, node, dir, x, id):
        """Merge vertex x into vertex 0 ('up': same side, 'down': opposite side)."""
        sign = 1 if dir == 'up' else -1
        Q = np.copy(node.Q)
        Q[:, 0] += sign * Q[:, x]
        Q[0, :] += sign * Q[x, :]
        Q = np.delete(np.delete(Q, x, axis=0), x, axis=1)

        Xfrac = np.delete(np.delete(node.Xfrac, x, axis=0), x, axis=1)

        sdp_mult = node.sdp_mult @ node.sdp_mult.T
        sdp_mult[:, 0] += sign * sdp_mult[:, x]
        sdp_mult[0, :] += sign * sdp_mult[x, :]
        sdp_mult = np.delete(np.delete(sdp_mult, x, axis=0), x, axis=1)
        C, D = np.linalg.eigh(sdp_mult)
        C = np.sqrt(np.maximum(C, 0))
        sdp_mult = D * C.reshape((1, -1))

        vm = np.copy(node.vertex_map)
        vs = np.copy(node.vertex_sign)
        eq_x = (vm == x); gt_x = (vm > x)
        vm[eq_x] = 0;  vs[eq_x] *= sign
        vm[gt_x] -= 1

        m = lambda v: 0 if v == x else (v - 1 if v > x else v)

        if dir == 'up':
            tri = None
            if len(node.tri) > 0:
                I  = [e for e, (i, j, k) in enumerate(node.tri)
                      if len(set((m(i), m(j), m(k)))) == 3]
                t  = [(m(node.tri[e][0]), m(node.tri[e][1]), m(node.tri[e][2])) for e in I]
                tm = [node.tri_mult[e] for e in I]
                seen = set(); I2 = []
                for idx, tri_ in enumerate(t):
                    if tri_ not in seen: seen.add(tri_); I2.append(idx)
                t = [t[i] for i in I2]; tm = [tm[i] for i in I2]
                tri = dict(mult=tm, tri=t)

            tri2 = None
            if len(node.tri2) > 0:
                I  = [e for e, (a, b, c) in enumerate(node.tri2)
                      if len(set((m(a), m(b), m(c)))) == 3]
                t  = [(m(node.tri2[e][0]), m(node.tri2[e][1]), m(node.tri2[e][2])) for e in I]
                tm = [node.tri2_mult[e] for e in I]
                seen = set(); I2 = []
                for idx, tri_ in enumerate(t):
                    if tri_ not in seen: seen.add(tri_); I2.append(idx)
                t = [t[i] for i in I2]; tm = [tm[i] for i in I2]
                tri2 = dict(mult=tm, tri=t)

        else:  
            new_tri  = []; new_tri_mult  = []
            new_tri2 = []; new_tri2_mult = []

            for e, (i, j, k) in enumerate(node.tri):
                mi, mj, mk = m(i), m(j), m(k)
                if i == x:
                    if len(set((0, mj, mk))) == 3:
                        new_tri2.append((0, mj, mk)); new_tri2_mult.append(node.tri_mult[e])
                elif j == x:
                    if len(set((0, mi, mk))) == 3:
                        new_tri2.append((0, mi, mk)); new_tri2_mult.append(node.tri_mult[e])
                elif k == x:
                    if len(set((0, mi, mj))) == 3:
                        new_tri2.append((0, mi, mj)); new_tri2_mult.append(node.tri_mult[e])
                else:
                    if len(set((mi, mj, mk))) == 3:
                        new_tri.append((mi, mj, mk)); new_tri_mult.append(node.tri_mult[e])

            for e, (a, b, c) in enumerate(node.tri2):
                ma, mb, mc = m(a), m(b), m(c)
                if a == x:
                    if len(set((0, mb, mc))) == 3:
                        new_tri.append((0, mb, mc)); new_tri_mult.append(node.tri2_mult[e])
                elif b == x:
                    if len(set((mc, ma, 0))) == 3:
                        new_tri2.append((mc, ma, 0)); new_tri2_mult.append(node.tri2_mult[e])
                elif c == x:
                    if len(set((mb, ma, 0))) == 3:
                        new_tri2.append((mb, ma, 0)); new_tri2_mult.append(node.tri2_mult[e])
                else:
                    if len(set((ma, mb, mc))) == 3:
                        new_tri2.append((ma, mb, mc)); new_tri2_mult.append(node.tri2_mult[e])

            seen = set(); t  = []; tm  = []
            for t_, mu_ in zip(new_tri, new_tri_mult):
                if t_ not in seen: seen.add(t_); t.append(t_); tm.append(mu_)
            tri = dict(mult=tm, tri=t)

            seen = set(); t2 = []; tm2 = []
            for t_, mu_ in zip(new_tri2, new_tri2_mult):
                if t_ not in seen: seen.add(t_); t2.append(t_); tm2.append(mu_)
            tri2 = dict(mult=tm2, tri=t2)

        child = maxcut(Q=Q, id=id, level=node.level + 1, sdp_mult=sdp_mult, tri=tri, tri2=tri2,
                       get_ub=node.external_ub, scale=node.scale, Xfrac=Xfrac,
                       vertex_map=vm, vertex_sign=vs)
        child.lb = node.lb
        return child

    def collapse(self):
        sign_of = {}
        rep_of = {}
        for start in range(self.n):
            if start in rep_of:
                continue
            sign_of[start] = 1
            rep_of[start] = start
            stack = [start]
            while stack:
                v = stack.pop()
                for j in range(self.n):
                    if j in rep_of:
                        continue
                    if self.fixed_mask[v, j]:
                        val = int(self.fixed_val_mat[v, j])
                    elif self.fixed_mask[j, v]:
                        val = int(self.fixed_val_mat[j, v])
                    else:
                        continue
                    sign_of[j] = sign_of[v] * val
                    rep_of[j] = rep_of[start]
                    stack.append(j)

        if all(rep_of[v] == v for v in range(self.n)):
            self.fixed_mask[:] = False
            self.fixed_val_mat[:] = 0
            return

        Q = np.copy(self.Q)
        L = self.sdp_mult @ self.sdp_mult.T
        for v in range(self.n):
            if rep_of[v] == v:
                continue
            r = rep_of[v]
            s = sign_of[v]
            Q[:, r] += s * Q[:, v]
            Q[r, :] += s * Q[v, :]
            L[:, r] += s * L[:, v]
            L[r, :] += s * L[v, :]

        keep = sorted(set(rep_of.values()))
        ix = np.ix_(keep, keep)
        Q = Q[ix]
        L = L[ix]
        Xfrac = self.Xfrac[ix]
        C, D = np.linalg.eigh(L)
        sdp_mult = D * np.sqrt(np.maximum(C, 0)).reshape((1, -1))
        new_idx = {keep[i]: i for i in range(len(keep))}

        agg_tri = {}; agg_tri2 = {}

        def add_tri(a, b, c, mu):
            k_ = tuple(sorted((a, b, c)))
            agg_tri[k_] = agg_tri.get(k_, 0.0) + mu

        def add_tri2(a, b, c, mu):
            k_ = (a, b, c) if b < c else (a, c, b)
            agg_tri2[k_] = agg_tri2.get(k_, 0.0) + mu

        for e, (va, vb, vc) in enumerate(self.tri):
            ra, rb, rc = rep_of[va], rep_of[vb], rep_of[vc]
            na, nb, nc = new_idx[ra], new_idx[rb], new_idx[rc]
            if len({na, nb, nc}) < 3:
                continue
            sa, sb, sc = sign_of[va], sign_of[vb], sign_of[vc]
            p, q = sa * sb, sa * sc
            if p == 1 and q == 1:
                add_tri(na, nb, nc, self.tri_mult[e])
            elif p == -1 and q == -1:
                add_tri2(na, nb, nc, self.tri_mult[e])
            elif p == -1:  # q == 1
                add_tri2(nb, na, nc, self.tri_mult[e])
            else:  # p == 1, q == -1
                add_tri2(nc, na, nb, self.tri_mult[e])

        for e, (va, vb, vc) in enumerate(self.tri2):
            ra, rb, rc = rep_of[va], rep_of[vb], rep_of[vc]
            na, nb, nc = new_idx[ra], new_idx[rb], new_idx[rc]
            if len({na, nb, nc}) < 3:
                continue
            sa, sb, sc = sign_of[va], sign_of[vb], sign_of[vc]
            p, q = sa * sb, sa * sc
            if p == 1 and q == 1:
                add_tri2(na, nb, nc, self.tri2_mult[e])
            elif p == -1 and q == -1:
                add_tri(na, nb, nc, self.tri2_mult[e])
            elif p == 1:  # q == -1
                add_tri2(nb, na, nc, self.tri2_mult[e])
            else:  # p == -1, q == 1
                add_tri2(nc, na, nb, self.tri2_mult[e])

        t  = list(agg_tri);  tm  = [agg_tri[k_]  for k_ in t]
        t2 = list(agg_tri2); tm2 = [agg_tri2[k_] for k_ in t2]

        rep_arr  = np.array([rep_of[i]  for i in self.vertex_map])
        sign_arr = np.array([sign_of[i] for i in self.vertex_map])
        self.vertex_map  = np.array([new_idx[r] for r in rep_arr])
        self.vertex_sign = self.vertex_sign * sign_arr

        self.n = len(keep)
        self.Q = Q
        self.sdp_mult = sdp_mult
        self.Xfrac = Xfrac
        self.X = np.empty_like(Q)
        self.C = np.empty_like(Q)
        self._QQT = np.empty_like(Q)
        self.tri = t
        self.tri_mult = np.array(tm) if tm else []
        self.tri2 = t2
        self.tri2_mult = np.array(tm2) if tm2 else []
        self.recalculate_indices_tri()
        self.recalculate_indices_tri2()
        self.fixed_mask = np.zeros((self.n, self.n), dtype=bool)
        self.fixed_val_mat = np.zeros((self.n, self.n), dtype=Q.dtype)
        self.candidate_mask = np.triu(np.ones((self.n, self.n), dtype=bool), k=1)

    def branch_3(self):
        X = self.Xfrac

        candidates = sorted([(i,j) for i in range(self.n) for j in range(i+1, self.n)], key=lambda x: abs(X[x[0],x[1]]), reverse=True)

        if len(candidates) == 0:
            return []

        x,y = candidates[0]


        def create_child(dir, x, y, id):
            Q = np.copy(self.Q)

            s = 1 if dir == 'up' else -1

            Q[x,x] += Q[y,y]
            Q[x,x] += s*Q[x,y]
            Q[x,x] += s*Q[y,x]
            Q[y,y] = 0
            Q[x,y] = 0
            Q[y,x] = 0
            Q[:,x] += s*Q[:,y]
            Q[x,:] += s*Q[y,:]
            Q = np.delete(np.delete(Q, y, axis=0), y, axis=1)

            #ONLY WORKS FOR NON PROJECTED
            sdp_mult = self.sdp_mult@self.sdp_mult.T
            sdp_mult[x,x] += sdp_mult[y,y]
            sdp_mult[x,x] += s*sdp_mult[x,y]
            sdp_mult[x,x] += s*sdp_mult[y,x]
            sdp_mult[y,y] = 0
            sdp_mult[x,y] = 0
            sdp_mult[y,x] = 0
            sdp_mult[:,x] += s*sdp_mult[:,y]
            sdp_mult[x,:] += s*sdp_mult[y,:]
            sdp_mult = np.delete(np.delete(sdp_mult, y, axis=0), y, axis=1)

            C, D = np.linalg.eigh(sdp_mult)
            C = np.sqrt(np.maximum(C, 0))
            sdp_mult = D*C.reshape((1,-1))

            m = lambda z: z-1 if z > y else z
            t = []
            tm = []
            t2 = []
            tm2 = []
            if len(self.tri) > 0:
                for (i,j,k), mu in zip(self.tri, self.tri_mult):
                    if i == y:
                        if x not in (i,j,k):
                            t2.append((m(x),m(j),m(k)))
                            tm2.append(mu)
                    elif j == y:
                        if x not in (i,j,k):
                            t2.append((m(x),m(i),m(k)))
                            tm2.append(mu)
                    elif k == y:
                        if x not in (i,j,k):
                            t2.append((m(x),m(i),m(j)))
                            tm2.append(mu)
                    else:
                        t.append((m(i),m(j),m(k)))
                        tm.append(mu)

            if len(self.tri2) > 0:
                for (i,j,k), mu in zip(self.tri2, self.tri2_mult):
                    if i == y:
                        if x not in (i,j,k):
                            t.append((m(x),m(j),m(k)))
                            tm.append(mu)
                    elif j == y:
                        if x not in (i,j,k):
                            t2.append((m(k),m(i),m(x)))
                            tm2.append(mu)
                    elif k == y:
                        if x not in (i,j,k):
                            t2.append((m(j),m(i),m(x)))
                            tm2.append(mu)
                    else:
                        t2.append((m(i),m(j),m(k)))
                        tm2.append(mu)
                      

            if len(t) > 0:
                tri = dict(mult=tm, tri=t)
            else:
                tri = None
            if len(t2) > 0:
                tri2 = dict(mult=tm2, tri=t2)
            else:
                tri2 = None

            child = maxcut(Q=Q, id=id, level=self.level+1, sdp_mult=sdp_mult, tri=tri, tri2=tri2, get_ub=self.external_ub, scale=self.scale)
            child.lb = self.lb

            return child

        nup = create_child('up', x, y, self.id*2)
        ndown = create_child('down', x, y, self.id*2+1)

       
        return [ndown, nup]

    def branch_4(self):
        def collapse(x, y, s, node):
            Q = np.copy(node.Q)

            Q[x,x] += Q[y,y]
            Q[x,x] += s*Q[x,y]
            Q[x,x] += s*Q[y,x]
            Q[y,y] = 0
            Q[x,y] = 0
            Q[y,x] = 0
            Q[:,x] += s*Q[:,y]
            Q[x,:] += s*Q[y,:]
            Q = np.delete(np.delete(Q, y, axis=0), y, axis=1)

            Xfrac = np.delete(np.delete(node.Xfrac, y, axis=0), y, axis=1)


            #ONLY WORKS FOR NON PROJECTED
            sdp_mult = node.sdp_mult@node.sdp_mult.T
            sdp_mult[x,x] += sdp_mult[y,y]
            sdp_mult[x,x] += s*sdp_mult[x,y]
            sdp_mult[x,x] += s*sdp_mult[y,x]
            sdp_mult[y,y] = 0
            sdp_mult[x,y] = 0
            sdp_mult[y,x] = 0
            sdp_mult[:,x] += s*sdp_mult[:,y]
            sdp_mult[x,:] += s*sdp_mult[y,:]
            sdp_mult = np.delete(np.delete(sdp_mult, y, axis=0), y, axis=1)

            C, D = np.linalg.eigh(sdp_mult)
            C = np.sqrt(np.maximum(C, 0))
            sdp_mult = D*C.reshape((1,-1))

            fixed_mask = np.copy(node.fixed_mask)
            fixed_val_mat = np.copy(node.fixed_val_mat)
            fixed_mask = np.delete(np.delete(fixed_mask, y, axis=0), y, axis=1)
            fixed_val_mat = np.delete(np.delete(fixed_val_mat, y, axis=0), y, axis=1)

            m = lambda z: z-1 if z > y else z
            t = []
            tm = []
            t2 = []
            tm2 = []
            if len(node.tri) > 0:
                for (i,j,k), mu in zip(node.tri, node.tri_mult):
                    if i == y:
                        if x not in (i,j,k):
                            t2.append((m(x),m(j),m(k)))
                            tm2.append(mu)
                    elif j == y:
                        if x not in (i,j,k):
                            t2.append((m(x),m(i),m(k)))
                            tm2.append(mu)
                    elif k == y:
                        if x not in (i,j,k):
                            t2.append((m(x),m(i),m(j)))
                            tm2.append(mu)
                    else:
                        t.append((m(i),m(j),m(k)))
                        tm.append(mu)

            if len(node.tri2) > 0:
                for (i,j,k), mu in zip(node.tri2, node.tri2_mult):
                    if i == y:
                        if x not in (i,j,k):
                            t.append((m(x),m(j),m(k)))
                            tm.append(mu)
                    elif j == y:
                        if x not in (i,j,k):
                            t2.append((m(k),m(i),m(x)))
                            tm2.append(mu)
                    elif k == y:
                        if x not in (i,j,k):
                            t2.append((m(j),m(i),m(x)))
                            tm2.append(mu)
                    else:
                        t2.append((m(i),m(j),m(k)))
                        tm2.append(mu)
                      

            if len(t) > 0:
                tri = dict(mult=tm, tri=t)
            else:
                tri = None
            if len(t2) > 0:
                tri2 = dict(mult=tm2, tri=t2)
            else:
                tri2 = None

            child = maxcut(Q=Q, id=node.id, level=node.level, sdp_mult=sdp_mult, tri=tri, tri2=tri2, get_ub=node.external_ub, scale=node.scale, Xfrac=Xfrac)
            child.lb = node.lb

            return child

        node = self
        col = True
        while col:
            col = False
            for i in range(node.n):
                if col: break
                for j in range(i+1, node.n):
                    if node.fixed_mask[i,j]:
                        col = True
                        print('collapsing', i, j)
                        node = collapse(i, j, node.fixed_val_mat[i,j], node)
                        break

        if node.n != self.n:
            node.solve_node()
            print(self.lb, node.lb)
            if node.get_ub()[0] is not None and np.ceil(node.lb-1e-5) >= round(node.get_ub()[0]):
                return []

        X = node.Xfrac

        candidates = sorted([(i,j) for i in range(1,node.n) for j in range(i+1, node.n)], key=lambda x: abs(X[x[0],x[1]]), reverse=True)

        if len(candidates) == 0:
            return []

        x,y = candidates[0]

        if y > x:
            y-=1

        nup = self._create_child(node, 'up', x, node.id*2)
        nupup = self._create_child(nup, 'up', y, nup.id*2)
        nupdown = self._create_child(nup, 'down', y, nup.id*2+1)
        ndown = self._create_child(node, 'down', x, node.id*2+1)
        ndownup = self._create_child(ndown, 'up', y, ndown.id*2)
        ndowndown = self._create_child(ndown, 'down', y, ndown.id*2+1)


       
        return [nupup, nupdown, ndownup, ndowndown]




    def branch_2(self):
        X = self.Xfrac

        candidates = sorted([(i,j) for i in range(1,self.n) for j in range(i+1, self.n)], key=lambda p: abs(X[p[0],p[1]]), reverse=True)

        if len(candidates) == 0:
            return []

        i, j = candidates[0]
        xij = X[i, j]



        if j > i:
            j -= 1

        nup = self._create_child(self, 'up', i, self.id*2)
        nupup = self._create_child(nup, 'up', j, nup.id*2)
        nupdown = self._create_child(nup, 'down', j, nup.id*2+1)
        ndown = self._create_child(self, 'down', i, self.id*2+1)
        ndownup = self._create_child(ndown, 'up', j, ndown.id*2)
        ndowndown = self._create_child(ndown, 'down', j, ndown.id*2+1)

        if xij > 0:
            return [nupup, ndowndown, ndownup, nupdown]
        else:
            return [nupdown, ndownup, nupup, ndowndown]

    def branch_1(self):
        X = self.Xfrac

        b = np.abs(np.mean(X*X[0,:], axis=1))

        candidates = sorted(list(range(1, self.n)), key=lambda v: b[v], reverse=True)

        if len(candidates) == 0:
            return []

        x = candidates[0]


        nup = self._create_child(self, 'up', x, self.id*2)
        ndown = self._create_child(self, 'down', x, self.id*2+1)

        return [ndown, nup]

    def _cosine_branch(self, pick_max):
        X = self.Xfrac

        norms = np.linalg.norm(X, axis=0)    
        safe  = np.where(norms > 1e-10, norms, 1.0)
        Xn    = X / safe[np.newaxis, :]     
        cos   = Xn[:, 0] @ Xn              

        interior = list(range(1, self.n))
        if not interior:
            return []

        key = lambda v: abs(cos[v])
        x = max(interior, key=key) if pick_max else min(interior, key=key)

        nup   = self._create_child(self, 'up',   x, self.id*2)
        ndown = self._create_child(self, 'down', x, self.id*2+1)

        if cos[x] > 0:
            return [ndown, nup]  
        else:
            return [nup, ndown] 

    def branch_5(self):
        return self._cosine_branch(pick_max=True)

    def branch_6(self):
        return self._cosine_branch(pick_max=False)

    def _delta_pair_branch(self, pick_max):
        deltas = -2.0 * self.X * self._QQT

        candidates = [(i, j) for i in range(1, self.n) for j in range(i+1, self.n)]
        if not candidates:
            return []

        i, j = max(candidates, key=lambda p: deltas[p[0], p[1]]) if pick_max \
               else min(candidates, key=lambda p: deltas[p[0], p[1]])
        xij = self.Xfrac[i, j]

        if j > i:
            j -= 1

        nup       = self._create_child(self,  'up',   i, self.id*2)
        nupup     = self._create_child(nup,   'up',   j, nup.id*2)
        nupdown   = self._create_child(nup,   'down', j, nup.id*2+1)
        ndown     = self._create_child(self,  'down', i, self.id*2+1)
        ndownup   = self._create_child(ndown, 'up',   j, ndown.id*2)
        ndowndown = self._create_child(ndown, 'down', j, ndown.id*2+1)

        if xij > 0:
            return [nupup, ndowndown, ndownup, nupdown]
        else:
            return [nupdown, ndownup, nupup, ndowndown]

    def branch_7(self):
        return self._delta_pair_branch(pick_max=True)

    def branch_8(self):
        return self._delta_pair_branch(pick_max=False)

    def branch_9(self):
        X = self.Xfrac
        candidates = sorted(
            [(i, j) for i in range(1, self.n) for j in range(i+1, self.n)],
            key=lambda p: abs(X[p[0], p[1]]), reverse=True
        )
        if not candidates:
            return []

        sb_iter = self.conf['sb_iter']
        sb_k    = self.conf['sb_k']

        def quick_conf(node):
            return {**node.conf,
                    'max_iter': sb_iter,
                    'frac':     False,
                    'cuts': {**node.conf['cuts'], 'add': False, 'rem': False}}

        best_score = -np.inf
        best_pair  = candidates[0]

        for i, j in candidates[:sb_k]:
            j_ = j - 1 if j > i else j   # j index after merging i

            t_up   = self._create_child(self,  'up',   i, -1)
            t_down = self._create_child(self,  'down', i, -1)

            grandchildren = [
                self._create_child(t_up,   'up',   j_, -1),
                self._create_child(t_up,   'down', j_, -1),
                self._create_child(t_down, 'up',   j_, -1),
                self._create_child(t_down, 'down', j_, -1),
            ]
            for gc in grandchildren:
                gc.conf = quick_conf(gc)
                gc.lagrangian_relaxation()

            score = min(gc.lb for gc in grandchildren)
            if score > best_score:
                best_score = score
                best_pair  = (i, j)

        i, j = best_pair
        xij  = X[i, j]

        if j > i:
            j -= 1

        nup       = self._create_child(self,  'up',   i, self.id*2)
        nupup     = self._create_child(nup,   'up',   j, nup.id*2)
        nupdown   = self._create_child(nup,   'down', j, nup.id*2+1)
        ndown     = self._create_child(self,  'down', i, self.id*2+1)
        ndownup   = self._create_child(ndown, 'up',   j, ndown.id*2)
        ndowndown = self._create_child(ndown, 'down', j, ndown.id*2+1)

        if xij > 0:
            return [nupup, ndowndown, ndownup, nupdown]
        else:
            return [nupdown, ndownup, nupup, ndowndown]

    def branch(self):
        return getattr(self, self.conf['branch'])()


