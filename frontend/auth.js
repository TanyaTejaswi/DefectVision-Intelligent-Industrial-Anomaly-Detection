/**
 * DEFECTVISION 2.0 - OPERATOR AUTHENTICATION & SESSION MANAGER
 * Manages Operator Logins, Workstation Tokens, Roles, and Permissions.
 */

const AuthManager = {
  currentUser: {
    username: 'admin',
    role: 'QA Operations Lead',
    workstationId: 'WS-STATION-04',
    token: 'jwt_mock_defectvision_2026'
  },

  init() {
    const saved = localStorage.getItem('defectvision_operator');
    if (saved) {
      try {
        this.currentUser = JSON.parse(saved);
      } catch (e) {
        console.warn('Failed to parse saved operator session');
      }
    }
    this.updateUI();
  },

  login(username, password, remember = true) {
    // Authenticate with default or custom credentials
    const role = (username.toLowerCase() === 'admin') ? 'QA Operations Lead' : 'Conveyor Operator';
    this.currentUser = {
      username: username || 'admin',
      role: role,
      workstationId: 'WS-STATION-' + Math.floor(10 + Math.random() * 90),
      token: 'jwt_' + Math.random().toString(36).substring(2)
    };

    if (remember) {
      localStorage.setItem('defectvision_operator', JSON.stringify(this.currentUser));
    }
    this.updateUI();
    return true;
  },

  logout() {
    localStorage.removeItem('defectvision_operator');
    this.currentUser = {
      username: 'guest',
      role: 'Guest Inspector',
      workstationId: 'WS-UNASSIGNED',
      token: null
    };
    this.updateUI();
  },

  updateUI() {
    const navText = document.getElementById('navOperatorText');
    if (navText) {
      navText.innerText = `Operator: ${this.currentUser.username} (${this.currentUser.role.split(' ')[0]})`;
    }
  }
};

window.AuthManager = AuthManager;
