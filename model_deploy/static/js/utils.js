class Utils {
    constructor(errorMessageId) {
      this.errorMessageId = errorMessageId;
    }
  
    createFileFromUrl(fileName, url, onSuccess, onError) {
      let request = new XMLHttpRequest();
      request.open('GET', url, true);
      request.responseType = 'arraybuffer';
  
      request.onload = function () {
        if (request.status === 200) {
          let data = new Uint8Array(request.response);
          cv.FS_createDataFile('/', fileName, data, true, false, false);
          if (onSuccess) onSuccess();
        } else {
          if (onError) onError(request.status);
          else {
            let errorMessageElement = document.getElementById(this.errorMessageId);
            if (errorMessageElement) {
              errorMessageElement.innerHTML = 'Failed to load ' + fileName + ' from ' + url + '.';
            }
          }
        }
      };
  
      request.onerror = function () {
        if (onError) onError(request.status);
        else {
          let errorMessageElement = document.getElementById(this.errorMessageId);
          if (errorMessageElement) {
            errorMessageElement.innerHTML = 'Network error occurred while trying to load ' + fileName + ' from ' + url + '.';
          }
        }
      };
  
      request.send();
    }
  }
  