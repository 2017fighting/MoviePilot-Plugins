import { createApp } from 'vue'
import App from './App.vue'
import DialogCloseBtn from './@core/components/DialogCloseBtn.vue'
import LoadingBanner from './@core/components/LoadingBanner.vue'
import PageContentTitle from './@core/components/PageContentTitle.vue'

const app = createApp(App);
// 5. 注册全局组件
app
  .component('VDialogCloseBtn', DialogCloseBtn)
  .component('LoadingBanner', LoadingBanner)
  .component('VPageContentTitle', PageContentTitle);
app.mount('#app');