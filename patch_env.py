p = "envs/flatland_env.py"
s = open(p, encoding="utf-8").read()

old_ret = (
'        terminated_all = all(dones[i] for i in range(self.n_agents)) or dones.get("__all__", False)\n'
'        done_dict["__all__"] = bool(terminated_all)\n'
'        trunc = self._episode_steps >= self.max_steps\n'
'        truncated_dict = {aid: bool(trunc) for aid in done_dict if aid != "__all__"}\n'
'        truncated_dict["__all__"] = bool(trunc)\n'
'        return obs_dict, reward_dict, done_dict, truncated_dict, info_dict'
)

new_ret = (
'        terminated_all = all(dones[i] for i in range(self.n_agents)) or dones.get("__all__", False)\n'
'        trunc = self._episode_steps >= self.max_steps\n'
'        if not hasattr(self, "_done_emitted"):\n'
'            self._done_emitted = set()\n'
'        active_obs, active_rew, term, truncd = {}, {}, {}, {}\n'
'        for i in range(self.n_agents):\n'
'            aid = f"agent_{i}"\n'
'            if aid in self._done_emitted:\n'
'                continue\n'
'            is_done = bool(dones[i])\n'
'            active_obs[aid] = obs_dict[aid]\n'
'            active_rew[aid] = reward_dict[aid]\n'
'            term[aid] = is_done\n'
'            truncd[aid] = bool(trunc)\n'
'            if is_done or trunc:\n'
'                self._done_emitted.add(aid)\n'
'        term["__all__"] = bool(terminated_all)\n'
'        truncd["__all__"] = bool(trunc)\n'
'        return active_obs, active_rew, term, truncd, {aid: {} for aid in active_obs}'
)

if old_ret in s:
    s = s.replace(old_ret, new_ret)
    print("OK: step() patche")
else:
    print("ATTENTION: bloc de retour non trouve")

marker = 'self._arrived = {f"agent_{i}": False for i in range(self.n_agents)}'
if marker in s and "_done_emitted = set()" not in s.split("def step")[0]:
    s = s.replace(marker, marker + "\n        self._done_emitted = set()", 1)
    print("OK: reset() initialise _done_emitted")

open(p, "w", encoding="utf-8").write(s)
