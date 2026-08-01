document.addEventListener('DOMContentLoaded', () => {
  const navbar = document.querySelector('.navbar');
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
        }
      });
    },
    { threshold: 0.1 }
  );

  document.querySelectorAll('.card, .case-card, .rule, .smc-item').forEach((el) => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(20px)';
    el.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
    observer.observe(el);
  });

  const style = document.createElement('style');
  style.textContent = '.visible { opacity: 1 !important; transform: translateY(0) !important; }';
  document.head.appendChild(style);

  window.addEventListener('scroll', () => {
    navbar.style.background =
      window.scrollY > 50
        ? 'rgba(10, 14, 23, 0.95)'
        : 'rgba(10, 14, 23, 0.85)';
  });
});
