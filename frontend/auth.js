/**
 * DEFECTVISION 2.0 - OPERATOR AUTHENTICATION & WORKSTATION SESSION MANAGER
 * Enforces authenticated access for component inspection, image selection, and defect logging.
 * Manages User ID, Work Email, Access Password/PIN, and Password Reset workflows.
 */

const DEFAULT_ACCOUNTS = [
  {
    userId: 'admin',
    email: 'admin@defectvision.industrial',
    password: 'DefectVision@Admin2026!',
    role: 'QA Operations Lead',
    workstationId: 'WS-101'
  },
  {
    userId: 'inspector_01',
    email: 'inspector@defectvision.industrial',
    password: 'Inspector123!',
    role: 'Conveyor Quality Inspector',
    workstationId: 'WS-102'
  },
  {
    userId: 'tanya_tejaswi',
    email: 'tanya@defectvision.industrial',
    password: 'VisionLead2026!',
    role: 'Computer Vision & Feature Fusion Lead',
    workstationId: 'WS-103'
  }
];

const AuthManager = {
  currentUser: null, // Null by default - requires login or account creation

  init() {
    // 1. Ensure accounts store exists
    if (!localStorage.getItem('defectvision_accounts')) {
      localStorage.setItem('defectvision_accounts', JSON.stringify(DEFAULT_ACCOUNTS));
    }

    // 2. Load saved session if exists
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

  getAccounts() {
    try {
      const stored = localStorage.getItem('defectvision_accounts');
      if (stored) {
        const list = JSON.parse(stored);
        if (Array.isArray(list)) return list;
      }
    } catch (e) {}
    return [...DEFAULT_ACCOUNTS];
  },

  saveAccounts(accounts) {
    try {
      localStorage.setItem('defectvision_accounts', JSON.stringify(accounts));
    } catch (e) {}
  },

  isAuthenticated() {
    return this.currentUser !== null && !!this.currentUser.token;
  },

  login(identifier, password, remember = true) {
    const cleanId = (identifier || '').trim();
    const cleanPass = (password || '').trim();
    if (!cleanId || !cleanPass) {
      alert('Please enter both your User ID / Email and password.');
      return false;
    }
    const accounts = this.getAccounts();

    // Match by User ID OR Email (case-insensitive)
    let account = accounts.find(a => 
      a.userId.toLowerCase() === cleanId.toLowerCase() || 
      (a.email && a.email.toLowerCase() === cleanId.toLowerCase())
    );

    if (!account) {
      alert('No account found for "' + cleanId + '". Please click "Create Account" below to register.');
      return false;
    }

    if (account.password && cleanPass !== account.password) {
      alert('Incorrect password. Please verify your password or click "Forgot Password".');
      return false;
    }

    this.currentUser = {
      username: account.userId,
      email: account.email,
      role: account.role,
      workstationId: account.workstationId || ('WS-' + Math.floor(100 + Math.random() * 900)),
      token: 'jwt_dv_' + Math.random().toString(36).substring(2) + Date.now().toString(36)
    };

    if (remember) {
      localStorage.setItem('defectvision_operator', JSON.stringify(this.currentUser));
    }
    this.updateUI();
    return true;
  },

  register(username, email, password, role = 'Conveyor Quality Inspector', remember = true) {
    const cleanUser = (username || 'operator_new').trim().replace(/\s+/g, '_');
    const cleanEmail = (email || `${cleanUser.toLowerCase()}@defectvision.industrial`).trim();
    const cleanPass = (password || 'Inspector123!').trim();
    const accounts = this.getAccounts();

    // Check if account already exists
    const existingIndex = accounts.findIndex(a => 
      a.userId.toLowerCase() === cleanUser.toLowerCase() || 
      (a.email && a.email.toLowerCase() === cleanEmail.toLowerCase())
    );

    const newAccount = {
      userId: cleanUser,
      email: cleanEmail,
      password: cleanPass,
      role: role || 'Conveyor Quality Inspector',
      workstationId: 'WS-' + Math.floor(100 + Math.random() * 900)
    };

    if (existingIndex >= 0) {
      accounts[existingIndex] = newAccount;
    } else {
      accounts.push(newAccount);
    }
    this.saveAccounts(accounts);

    this.currentUser = {
      username: newAccount.userId,
      email: newAccount.email,
      role: newAccount.role,
      workstationId: newAccount.workstationId,
      token: 'jwt_dv_' + Math.random().toString(36).substring(2) + Date.now().toString(36)
    };

    if (remember) {
      localStorage.setItem('defectvision_operator', JSON.stringify(this.currentUser));
    }
    this.updateUI();
    return true;
  },

  resetPassword(identifier, newPassword) {
    const cleanId = (identifier || '').trim();
    const cleanPass = (newPassword || 'NewPassword2026!').trim();
    const accounts = this.getAccounts();

    let account = accounts.find(a => 
      a.userId.toLowerCase() === cleanId.toLowerCase() || 
      (a.email && a.email.toLowerCase() === cleanId.toLowerCase())
    );

    if (account) {
      account.password = cleanPass;
    } else {
      account = {
        userId: cleanId.includes('@') ? cleanId.split('@')[0] : (cleanId || 'operator_reset'),
        email: cleanId.includes('@') ? cleanId : `${cleanId.toLowerCase()}@defectvision.industrial`,
        password: cleanPass,
        role: 'Conveyor Quality Inspector',
        workstationId: 'WS-' + Math.floor(100 + Math.random() * 900)
      };
      accounts.push(account);
    }
    this.saveAccounts(accounts);

    // Automatically authenticate the operator with the new credentials
    this.currentUser = {
      username: account.userId,
      email: account.email,
      role: account.role,
      workstationId: account.workstationId,
      token: 'jwt_dv_' + Math.random().toString(36).substring(2) + Date.now().toString(36)
    };

    localStorage.setItem('defectvision_operator', JSON.stringify(this.currentUser));
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
        navText.innerHTML = `
          <span class="w-2 h-2 rounded-full bg-[#00B050] inline-block mr-1.5 animate-pulse"></span>
          <span class="font-bold text-gray-900">${this.currentUser.username}</span> 
          <span class="text-gray-500 font-normal">(${this.currentUser.role.split(' ')[0]})</span> 
          <button onclick="event.stopPropagation(); AuthManager.logout()" class="text-[10px] text-gray-400 hover:text-red-600 underline ml-2 font-mono font-bold" title="Sign out of workstation">Sign out</button>
        `;
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

    // Synchronize Personal Inspection Audit Log to the active operator
    if (typeof window.initAuditLog === 'function') {
      window.initAuditLog();
    }
    if (typeof window.renderAuditLogPage === 'function' && window.state && window.state.currentPage === 'auditLog') {
      window.renderAuditLogPage();
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
