# NeuroSign - AI-Based Sign Language to Text Conversion System

A modern, responsive web application that converts sign language gestures into text using AI and computer vision. This is a comprehensive frontend for a college mini-project that demonstrates gesture recognition capabilities with an interactive demo interface.

## 🌟 Features

### User Interface
- **Modern & Responsive Design** - Fully responsive across desktop, tablet, and mobile devices
- **Green-Based Color Palette** - Professional and accessible design using emerald green as primary color
- **Smooth Animations** - Subtle transitions and hover effects throughout the application
- **Accessible** - Clean typography and high contrast for users with visual impairments

### Sections Included

1. **Navigation Bar**
   - Sticky navbar with smooth scroll navigation
   - Mobile hamburger menu for responsive design
   - Project logo and name

2. **Hero Section**
   - Large heading with gradient text effect
   - Compelling subtitle explaining the technology
   - Primary CTA: "Start Recognition"
   - Secondary CTA: "View Demo"
   - Animated visual element with gesture representation

3. **Live Recognition Demo**
   - Real-time webcam preview (simulated with placeholder)
   - Gesture recognition display with confidence score
   - Generated text output
   - Gesture history
   - Speech output functionality (text-to-speech)
   - Interactive camera controls

4. **How It Works**
   - Step-by-step workflow visualization
   - 7-stage pipeline: Input → Preprocessing → Feature Extraction → CNN Classification → Text Output → Speech Output
   - Clean icon-based design with descriptions

5. **Applications Section**
   - Showcase 6 key use cases:
     - Communication Aid for deaf/mute individuals
     - Educational Learning Tool
     - Healthcare & Public Services
     - Human-Computer Interaction
     - Smart Assistive Technology
     - Accessibility for All

6. **Future Scope**
   - 6 planned enhancements:
     - Dynamic Gesture Recognition
     - NLP-based Sentence Generation
     - Enhanced Text-to-Speech
     - Mobile App Support
     - Larger Vocabulary Support
     - Regional Sign Language Support

7. **Contact Section**
   - Contact information cards
   - Contact form for inquiries
   - Responsive layout

8. **Footer**
   - Project description
   - Team member credits
   - Guide information
   - Copyright notice

### Interactive Features
- **Live Demo Simulation** - Randomly generates sample gestures with confidence scores
- **Gesture History** - Tracks recently recognized gestures
- **Text-to-Speech** - Converts recognized text to speech using Web Speech API
- **Clear Functionality** - Reset text output and history
- **Form Submission** - Contact form with success feedback
- **Smooth Scrolling** - Enhanced user navigation

## 🎨 Design Features

### Color Scheme
- **Primary Green**: #10b981 (Emerald)
- **Dark Green**: #059669
- **Light Mint**: #ecfdf5
- **Mint Accent**: #d1fae5
- **Darker Green**: #047857

### Typography
- Clean, modern system fonts (SF Pro Display, Segoe UI, etc.)
- Professional hierarchy with clear visual distinction
- Optimized for readability

### Visual Elements
- Rounded cards with subtle shadows
- Gradient backgrounds in key sections
- Icon-based feature representation (no emojis)
- Smooth hover effects and transitions
- Responsive grid layouts

## 📁 Project Structure

```
neurosign/
├── index.html          # Main HTML file with all sections
├── styles.css          # Complete CSS styling with responsive design
├── script.js           # JavaScript for interactivity
└── README.md           # This file
```

## 🚀 Getting Started

### Prerequisites
- Modern web browser (Chrome, Firefox, Safari, Edge)
- No backend or server required for demo version

### Installation & Usage

1. **Clone or Download** the project files
2. **Open in Browser**:
   - Simply double-click `index.html`, or
   - Drag and drop `index.html` to your browser, or
   - Use a local development server:
     ```bash
     # Using Python 3
     python -m http.server 8000
     
     # Using Python 2
     python -m SimpleHTTPServer 8000
     
     # Using Node.js (if http-server is installed)
     http-server
     ```
3. **Navigate** to `http://localhost:8000` (if using server)

### Features to Try

- **Live Demo Section**:
  - Click "Start Camera" to begin gesture recognition simulation
  - Watch gestures appear with confidence scores
  - View generated text output
  - Click "Speak Text" to hear the text read aloud
  - Use "Clear Text" to reset the output

- **Navigation**:
  - Click navbar links to smoothly scroll to each section
  - Try the mobile menu hamburger on smaller screens

- **Contact Form**:
  - Fill in and submit the contact form
  - See success feedback message

## 💻 Technology Stack

- **HTML5** - Semantic markup
- **CSS3** - Modern styling with flexbox and grid
- **JavaScript (ES6+)** - Interactive features
- **Web APIs Used**:
  - Web Speech API for text-to-speech
  - Intersection Observer for animations
  - LocalStorage (ready for extension)

## 🎯 Functional Requirements Met

✅ Fully responsive for desktop, tablet, and mobile  
✅ Clean and accessible UI  
✅ Smooth transitions and hover effects  
✅ Placeholder data for live recognition  
✅ Suitable for college mini project presentation  
✅ Professional, assistive-tech oriented design  
✅ No external dependencies (standalone frontend)  

## 📱 Responsive Breakpoints

- **Desktop**: 1024px and above
- **Tablet**: 768px to 1023px
- **Mobile**: Below 768px
- **Small Mobile**: 480px and below

## 🔧 Customization

### Colors
Edit the CSS variables at the top of `styles.css`:
```css
:root {
    --primary-green: #10b981;
    --dark-green: #059669;
    /* ... other colors ... */
}
```

### Content
- Update team member names in the footer
- Modify guide name and affiliation
- Change contact information
- Add real backend endpoints when ready

### Gestures (Demo Data)
Edit the `sampleGestures` array in `script.js` to change:
- Gesture names
- Output text
- Default confidence scores

## 🔮 Future Enhancements

When ready to integrate the backend:
1. Replace simulated gesture recognition with actual ML model API calls
2. Connect to real webcam using WebRTC
3. Add user authentication
4. Implement database for gesture history
5. Add multi-language support
6. Create PWA for offline support
7. Integrate actual text-to-speech API

## 📊 Browser Compatibility

- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

## 🎓 College Project Notes

This frontend is designed specifically for:
- College mini project presentation
- Demonstration of gesture recognition capabilities
- Interactive live demo section
- Professional appearance suitable for academic showcase
- Easy to extend with backend ML models

## 🎨 Accessibility Features

- High contrast text and backgrounds
- Semantic HTML structure
- ARIA labels where appropriate
- Keyboard navigation support
- Mobile-friendly touch targets
- Clear visual feedback on interactions

## 📝 Notes for Developers

### Adding Real Gesture Recognition
To connect with actual ML backend:
```javascript
// Replace the simulated recognition with API call
async function recognizeGesture(frameData) {
    const response = await fetch('/api/recognize-gesture', {
        method: 'POST',
        body: frameData
    });
    const result = await response.json();
    updateRecognitionDisplay(result.gesture, result.text, result.confidence);
}
```

### Web Speech API
The text-to-speech feature uses the Web Speech API. Some browsers may require HTTPS in production.

## 📄 License

This project is created for educational purposes as a college mini project.

## 👥 Credits

Created as a demonstration of modern web development practices with focus on:
- Responsive design
- User experience
- Accessibility
- Interactive features
- Professional UI/UX

---

**Ready to demonstrate your AI-powered gesture recognition system!** 🚀

For questions or support regarding this frontend, refer to the inline comments in the HTML, CSS, and JavaScript files.
