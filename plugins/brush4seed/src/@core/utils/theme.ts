export function saveLocalTheme(name: string, theme: any) {
  // 存储主题到本地
  localStorage.setItem('theme', name)
  localStorage.setItem('materio-initial-loader-bg', theme.current.value.colors.background)
  localStorage.setItem('materio-initial-loader-color', theme.current.value.colors.primary)
}
