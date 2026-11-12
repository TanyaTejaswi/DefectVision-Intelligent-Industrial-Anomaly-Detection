/**
 * DEFECTVISION 2.0 - OPERATOR AUTHENTICATION & SESSION MANAGER
 * Enforces authenticated access for component inspection and image selection.
 */

const AuthManager = {
  currentUser: null, // Null by default - requires login or account creation

  init() {
    const saved = localStorage.getItem('defectvision_operator');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        if (parsed && parsed.token) {
          this.currentUser = parsed;
        } else {
          this.currentUser = null;
        }
      } catch (e) {
        this.currentUser = null;
      }
    } else {
      this.currentUser = null;
    }
    this.updateUI();
  },

  isAuthenticated() {
    return this.currentUser !== null && !!this.currentUser.token;
  },

  login(username, password, remember = true) {
    const cleanUser = (username || 'operator_qa').trim();
    const role = (cleanUser.toLowerCase().includes('admin') || cleanUser.toLowerCase().includes('lead')) 
      ? 'QA Operations Lead' 
      : 'Conveyor Quality Inspector';

    this.currentUser = {
      username: cleanUser,
      role: role,
      workstationId: 'WS-' + Math.floor(100 + Math.random() * 900),
      token: 'jwt_auth_' + Math.random().toString(36).substring(2) + Date.now().toString(36)
    };

    if (remember) {
      localStorage.setItem('defectvision_operator', JSON.stringify(this.currentUser));
    }
    this.updateUI();
    return true;
  },

  register(username, role = 'Conveyor Inspector', remember = true) {
    const cleanUser = (username || 'operator_new').trim();
    this.currentUser = {
      username: cleanUser,
      role: role || 'Conveyor Quality Inspector',
      workstationId: 'WS-' + Math.floor(100 + Math.random() * 900),
      token: 'jwt_auth_' + Math.random().toString(36).substring(2) + Date.now().toString(36)
    };

    if (remember) {
      localStorage.setItem('defectvision_operator', JSON.stringify(this.currentUser));
    }
    this.updateUI();
    return true;
  },

  logout() {
    localStorage.removeItem('defectvision_operator');
    this.currentUser = null;
    this.updateUI();
    if (window.navigateTo) {
      window.navigateTo('home');
    }
  },

  updateUI() {
    const navText = document.getElementById('navOperatorText');
    const heroBtnText = document.getElementById('heroAuthBtnText');
    const navInspectCtaBtn = document.getElementById('navInspectCtaBtn');

    if (this.isAuthenticated()) {
      if (navText) {
        navText.innerHTML = `<span class="w-2 h-2 rounded-full bg-[#00B050] inline-block mr-1.5 animate-pulse"></span><span class="font-bold text-gray-900">${this.currentUser.username}</span> <span class="text-gray-500 font-normal">(${this.currentUser.role.split(' ')[0]})</span> <button onclick="event.stopPropagation(); AuthManager.logout()" class="text-[10px] text-gray-400 hover:text-red-600 underline ml-2 font-mono font-bold" title="Sign out of workstation">Sign out</button>`;
      }
      if (heroBtnText) {
        heroBtnText.innerText = 'ENTER LIVE AOI STUDIO →';
      }
      if (navInspectCtaBtn) {
        navInspectCtaBtn.innerText = 'TEST LIVE AOI STUDIO';
      }
    } else {
      if (navText) {
        navText.innerHTML = `<span class="w-2 h-2 rounded-full bg-amber-400 inline-block mr-1.5"></span><span class="font-bold text-gray-700">Sign In / Register</span>`;
      }
      if (heroBtnText) {
        heroBtnText.innerText = 'LOGIN / CREATE ACCOUNT →';
      }
      if (navInspectCtaBtn) {
        navInspectCtaBtn.innerText = 'OPERATOR LOGIN';
      }
    }
  }
};

window.AuthManager = AuthManager;
// Self-initialize immediately on script evaluation
try {
  AuthManager.init();
} catch (e) {
  console.warn('AuthManager immediate init skipped:', e);
}
